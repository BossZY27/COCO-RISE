from sqlalchemy import select, func, text
from .models import (Chunk, CorpusRevision, Document, Embedding, EmbeddingModel, Generation,
                     Job, Version)
from .repository import revision


class VectorRepository:
    def __init__(self, database):
        self.db = database

    def ready(self, corpus_revision):
        with self.db.transaction() as s:
            return bool(s.scalar(select(Generation.id).where(Generation.corpus_revision == corpus_revision,
                                                            Generation.status == "active").limit(1)))

    def model_revision(self, corpus_revision):
        with self.db.transaction() as s:
            return s.scalar(select(EmbeddingModel.revision).join(Generation, Generation.model_id == EmbeddingModel.id)
                .where(Generation.corpus_revision == corpus_revision, Generation.status == "active")
                .order_by(Generation.created_at.desc()).limit(1))

    def search(self, vector, year, corpus_revision):
        with self.db.transaction() as s:
            generation = s.scalar(select(Generation.id).where(Generation.corpus_revision == corpus_revision,
                                 Generation.status == "active").order_by(Generation.created_at.desc()).limit(1))
            if generation is None:
                return []
            distance = Embedding.vector.cosine_distance(vector)
            query = (select(Chunk, Document, Version).join(Version, Chunk.version_id == Version.id)
                     .join(Document, Version.document_id == Document.id)
                     .join(Embedding, Embedding.chunk_id == Chunk.id)
                     .where(Embedding.generation_id == generation, Version.current.is_(True),
                            Version.status == "published", Chunk.verified.is_(True), distance < .65)
                     .order_by(distance).limit(12))
            if year:
                query = query.where((Document.fiscal_year_be.is_(None)) | (Document.fiscal_year_be == year))
            return [{**c.payload, "id": c.legacy_id or str(c.id), "text": c.text, "doc": d.title,
                     "fiscal_year": d.fiscal_year_be, "page": c.page, "verified": True, "status": "active",
                     "document_id": str(d.id), "version_id": str(v.id), "chunk_id": str(c.id)}
                    for c, d, v in s.execute(query)]

    def build(self, client):
        # No transaction is held while the model computes embeddings.
        digest = client.revision(client.settings.embed_model)
        with self.db.transaction() as s:
            rev = revision(s)
            rows = [(c.id, c.text) for c in s.scalars(select(Chunk).join(Version, Chunk.version_id == Version.id)
                    .where(Version.current.is_(True), Version.status == "published", Chunk.verified.is_(True)))]
            model = s.scalar(select(EmbeddingModel).where(EmbeddingModel.revision == digest))
            if model is None:
                model = EmbeddingModel(name=client.settings.embed_model, revision=digest)
                s.add(model)
                s.flush()
            generation = Generation(model_id=model.id, corpus_revision=rev)
            s.add(generation)
            s.flush()
            identifier = generation.id
        try:
            for start in range(0, len(rows), 16):
                batch = rows[start:start + 16]
                vectors = client.embed([value for _, value in batch], timeout=120)
                with self.db.transaction() as s:
                    s.add_all(Embedding(generation_id=identifier, chunk_id=cid, vector=v)
                              for (cid, _), v in zip(batch, vectors))
            with self.db.transaction() as s:
                # Serialize activation with publication/review revision changes.
                s.execute(select(CorpusRevision).where(CorpusRevision.id == 1).with_for_update())
                generation = s.get(Generation, identifier)
                count = s.scalar(select(func.count()).select_from(Embedding).where(Embedding.generation_id == identifier))
                if count != len(rows) or revision(s) != rev:
                    generation.status = "stale"
                    s.add(Job(kind="index"))
                    return False
                generation.status = "active"
            return True
        except Exception:
            with self.db.transaction() as s:
                s.get(Generation, identifier).status = "failed"
            raise
