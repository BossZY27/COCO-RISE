import httpx
import threading
import time
from app.modules.knowledge import corpus as store
from app.modules.knowledge.context import use_corpus


class HybridRetrieval:
    def __init__(self, vectors, embeddings, timeout):
        self.vectors, self.embeddings, self.timeout = vectors, embeddings, timeout
        self.failed_until = 0.0
        self.lock = threading.Lock()

    def search(self, question, year, corpus):
        with use_corpus(corpus):
            keyword = store.keyword_search(question, year, top_k=12)
            semantic = []
            method = "bm25"
            if time.monotonic() >= self.failed_until and self.vectors.ready(corpus.revision):
                try:
                    if self.embeddings.revision(self.embeddings.settings.embed_model) != self.vectors.model_revision(corpus.revision):
                        raise ValueError("Embedding model revision changed; rebuild the index")
                    vector = self.embeddings.embed([question], timeout=self.timeout)[0]
                    semantic = self.vectors.search(vector, year, corpus.revision)
                    method = "bm25+pgvector"
                except (OSError, ValueError, RuntimeError, httpx.HTTPError):
                    self.failed_until = time.monotonic() + 30
            return store.rrf([keyword, semantic], top_k=8), method
