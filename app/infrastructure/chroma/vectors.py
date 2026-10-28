"""ChromaDB vector index adapter paired with PostgreSQL relational tables.

PostgreSQL stores relational metadata, document versions, facts, rules, and audit logs.
ChromaDB (PersistentClient on disk at data/chroma_db) stores the 1,024-dimension bge-m3
embeddings and performs HNSW cosine similarity search.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from sqlalchemy import select

from app.infrastructure.postgres.models import (
    Chunk,
    CorpusRevision,
    Document,
    EmbeddingModel,
    Generation,
    Job,
    Version,
)
from app.infrastructure.postgres.repository import revision

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHROMA_DIR = ROOT / "data" / "chroma_db"
CORPUS_COLLECTION_NAME = "coco_rise_chunks"

_CLIENT_LOCK = threading.Lock()
_CLIENTS: dict[str, Any] = {}


def get_chroma_client(chroma_dir: Path | str | None = None):
    """Return a thread-safe PersistentClient for the given ChromaDB directory."""
    target = Path(chroma_dir) if chroma_dir else DEFAULT_CHROMA_DIR
    key = str(target.resolve())
    with _CLIENT_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            target.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=key,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            _CLIENTS[key] = client
        return client


def sync_corpus_chunks(
    chunks: list[dict],
    vectors_by_id: dict[str, list[float]],
    model_name: str = "bge-m3:latest",
    chroma_dir: Path | str | None = None,
) -> int:
    """Sync verified/draft corpus chunks and their bge-m3 vectors into ChromaDB."""
    if not vectors_by_id:
        return 0
    client = get_chroma_client(chroma_dir)
    with _CLIENT_LOCK:
        collection = client.get_or_create_collection(
            name=CORPUS_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embed_model": model_name},
        )
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for chunk in chunks:
            cid = str(chunk.get("id") or "")
            vec = vectors_by_id.get(cid)
            if not cid or not isinstance(vec, list) or not vec:
                continue
            ids.append(cid)
            embeddings.append([float(x) for x in vec])
            documents.append(str(chunk.get("text") or ""))
            metadatas.append(
                {
                    "doc": str(chunk.get("doc") or ""),
                    "fiscal_year": int(chunk["fiscal_year"]) if chunk.get("fiscal_year") is not None else 0,
                    "clause": str(chunk.get("clause") or ""),
                    "page": int(chunk.get("page") or 1),
                    "status": str(chunk.get("status") or "active"),
                    "verified": 1 if chunk.get("verified") is True else 0,
                    "embed_model": model_name,
                }
            )
        if not ids:
            return 0
        if collection.count() != len(ids):
            for start in range(0, len(ids), 64):
                collection.upsert(
                    ids=ids[start : start + 64],
                    embeddings=embeddings[start : start + 64],
                    documents=documents[start : start + 64],
                    metadatas=metadatas[start : start + 64],
                )
        return collection.count()


def search_corpus_chunks(
    query_vector: list[float],
    chunks_by_id: dict[str, dict],
    top_k: int = 12,
    chroma_dir: Path | str | None = None,
) -> list[tuple[float, dict]] | None:
    """Query the ChromaDB corpus collection by cosine similarity.

    Returns a list of (cosine_similarity, chunk_dict) or None if the collection
    is empty or has a different dimension (e.g. during synthetic unit tests).
    """
    try:
        client = get_chroma_client(chroma_dir)
        collection = client.get_or_create_collection(
            name=CORPUS_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        total = collection.count()
        if total == 0 or total != len(chunks_by_id):
            return None
        res = collection.query(
            query_embeddings=[[float(x) for x in query_vector]],
            n_results=min(max(top_k * 2, 12), total),
            include=["distances"],
        )
        id_lists = res.get("ids") or [[]]
        dist_lists = res.get("distances") or [[]]
        if not id_lists or not id_lists[0]:
            return []
        scored: list[tuple[float, dict]] = []
        for cid, dist in zip(id_lists[0], dist_lists[0]):
            chunk = chunks_by_id.get(cid)
            if chunk is None:
                continue
            # Chroma cosine distance = 1.0 - cosine_similarity
            similarity = 1.0 - float(dist)
            scored.append((similarity, chunk))
        return scored
    except Exception:
        return None


def collection_stats(chroma_dir: Path | str | None = None) -> dict[str, Any]:
    """Return ChromaDB collection statistics for the database inspector."""
    try:
        client = get_chroma_client(chroma_dir)
        collection = client.get_or_create_collection(
            name=CORPUS_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        count = collection.count()
    except Exception:
        count = 0
    return {
        "engine": "ChromaDB (PersistentClient · HNSW Cosine)",
        "collection": CORPUS_COLLECTION_NAME,
        "path": "data/chroma_db",
        "vector_count": count,
        "dimension": 1024,
        "model": "bge-m3:latest",
    }


class ChromaVectorRepository:
    """PostgreSQL + ChromaDB vector repository for the v2 runtime.

    PostgreSQL tracks CorpusRevision, EmbeddingModel, Generation, and Chunk/Version
    verification status; ChromaDB stores and searches the 1,024-dim vectors per generation.
    """

    def __init__(self, database, chroma_dir: Path | str | None = None):
        self.db = database
        self.chroma_dir = chroma_dir

    @staticmethod
    def _collection_name(generation_id) -> str:
        clean = str(generation_id).replace("-", "")
        return f"rise_gen_{clean[:48]}"

    def ready(self, corpus_revision):
        with self.db.transaction() as s:
            return bool(
                s.scalar(
                    select(Generation.id)
                    .where(
                        Generation.corpus_revision == corpus_revision,
                        Generation.status == "active",
                    )
                    .limit(1)
                )
            )

    def model_revision(self, corpus_revision):
        with self.db.transaction() as s:
            return s.scalar(
                select(EmbeddingModel.revision)
                .join(Generation, Generation.model_id == EmbeddingModel.id)
                .where(
                    Generation.corpus_revision == corpus_revision,
                    Generation.status == "active",
                )
                .order_by(Generation.created_at.desc())
                .limit(1)
            )

    def search(self, vector, year, corpus_revision):
        with self.db.transaction() as s:
            generation = s.scalar(
                select(Generation.id)
                .where(
                    Generation.corpus_revision == corpus_revision,
                    Generation.status == "active",
                )
                .order_by(Generation.created_at.desc())
                .limit(1)
            )
            if generation is None:
                return []

        try:
            client = get_chroma_client(self.chroma_dir)
            collection = client.get_collection(name=self._collection_name(generation))
            total = collection.count()
            if total == 0:
                return []
            res = collection.query(
                query_embeddings=[[float(x) for x in vector]],
                n_results=min(24, total),
                include=["distances"],
            )
        except Exception as exc:
            raise RuntimeError("Vector index unavailable") from exc

        hit_ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        qualified_ids = [
            cid for cid, dist in zip(hit_ids, distances) if float(dist) < 0.65
        ]
        if not qualified_ids:
            return []

        with self.db.transaction() as s:
            query = (
                select(Chunk, Document, Version)
                .join(Version, Chunk.version_id == Version.id)
                .join(Document, Version.document_id == Document.id)
                .where(
                    Version.current.is_(True),
                    Version.status == "published",
                    Chunk.verified.is_(True),
                )
            )
            if year:
                query = query.where(
                    (Document.fiscal_year_be.is_(None))
                    | (Document.fiscal_year_be == year)
                )
            by_chunk_id = {
                str(c.id): (c, d, v) for c, d, v in s.execute(query)
            }
            ordered = []
            for cid in qualified_ids:
                if cid in by_chunk_id:
                    c, d, v = by_chunk_id[cid]
                    ordered.append(
                        {
                            **c.payload,
                            "id": c.legacy_id or str(c.id),
                            "text": c.text,
                            "doc": d.title,
                            "fiscal_year": d.fiscal_year_be,
                            "page": c.page,
                            "verified": True,
                            "status": "active",
                            "document_id": str(d.id),
                            "version_id": str(v.id),
                            "chunk_id": str(c.id),
                        }
                    )
                    if len(ordered) >= 12:
                        break
            return ordered

    def build(self, client):
        # No database transaction is held while Ollama computes embeddings and ChromaDB indexes them.
        digest = client.revision(client.settings.embed_model)
        with self.db.transaction() as s:
            rev = revision(s)
            rows = [
                (c.id, c.text)
                for c in s.scalars(
                    select(Chunk)
                    .join(Version, Chunk.version_id == Version.id)
                    .where(
                        Version.current.is_(True),
                        Version.status == "published",
                        Chunk.verified.is_(True),
                    )
                )
            ]
            model = s.scalar(
                select(EmbeddingModel).where(EmbeddingModel.revision == digest)
            )
            if model is None:
                model = EmbeddingModel(
                    name=client.settings.embed_model, revision=digest
                )
                s.add(model)
                s.flush()
            generation = Generation(model_id=model.id, corpus_revision=rev)
            s.add(generation)
            s.flush()
            identifier = generation.id

        col_name = self._collection_name(identifier)
        try:
            chroma_client = get_chroma_client(self.chroma_dir)
            collection = chroma_client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine", "generation_id": str(identifier)},
            )
            for start in range(0, len(rows), 16):
                batch = rows[start : start + 16]
                vectors = client.embed([value for _, value in batch], timeout=120)
                collection.upsert(
                    ids=[str(cid) for cid, _ in batch],
                    embeddings=[[float(x) for x in v] for v in vectors],
                    documents=[text for _, text in batch],
                )
            with self.db.transaction() as s:
                s.execute(
                    select(CorpusRevision)
                    .where(CorpusRevision.id == 1)
                    .with_for_update()
                )
                generation = s.get(Generation, identifier)
                count = collection.count()
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
