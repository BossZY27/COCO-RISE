"""Compose ports and adapters once per process; imports do not open DB/model clients."""
from app.infrastructure.files.storage import FileStorage
from app.infrastructure.ollama.client import OllamaClient
from app.infrastructure.postgres.corpus import CorpusRepository
from app.infrastructure.postgres.repository import Repository
from app.infrastructure.postgres.uow import Database
from app.infrastructure.postgres.vectors import VectorRepository
from app.modules.query.service import QueryService
from app.modules.retrieval.service import HybridRetrieval


class Services:
    def __init__(self, settings):
        self.settings = settings
        self.database = Database(settings.database_url)
        self.files = FileStorage(settings.files_dir, settings.max_pdf_bytes, settings.max_pdf_pages)
        self.repository = Repository(self.database, self.files, settings)
        self.corpora = CorpusRepository(self.database)
        self.ollama = OllamaClient(settings)
        self.vectors = VectorRepository(self.database)
        self.retrieval = HybridRetrieval(self.vectors, self.ollama, settings.semantic_query_timeout)
        self.query = QueryService(self.corpora, self.repository, self.retrieval, self.ollama)

    def status(self):
        database = self.database.ready()
        try:
            tags = self.ollama.tags()
            names = {m.get("name") for m in tags}
            missing = [m for m in (self.settings.answer_model, self.settings.embed_model)
                       if m not in names and m + ":latest" not in names]
            models = "ready" if not missing else "missing: " + ", ".join(missing)
        except Exception:
            models = "unavailable"
        return {"status": "ok" if database and models == "ready" else "degraded",
                "database": "ready" if database else "migration_required", "models": models,
                "fallback": "verified_template+bm25"}

    def close(self):
        self.ollama.close()
        self.database.close()
