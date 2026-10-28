"""BM25 + BGE-M3 retrieval; degraded requests explicitly report BM25 fallback."""
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

    def search(self, question, year, corpus, timeout=None):
        with use_corpus(corpus):
            keyword = store.keyword_search(question, year, top_k=12)
            semantic, reason = [], None
            deadline = time.monotonic() + min(self.timeout, timeout if timeout is not None else self.timeout)
            with self.lock:
                cooling_down = time.monotonic() < self.failed_until
            if cooling_down:
                reason = "embedding_cooldown"
            elif not self.vectors.ready(corpus.revision):
                reason = "index_not_ready"
            else:
                try:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Embedding deadline reached")
                    digest = self.embeddings.revision(self.embeddings.settings.embed_model, timeout=min(1, remaining))
                    if digest != self.vectors.model_revision(corpus.revision):
                        raise ValueError("Embedding model revision changed; rebuild the index")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Embedding deadline reached")
                    vector = self.embeddings.embed([question], timeout=remaining)[0]
                    semantic = self.vectors.search(vector, year, corpus.revision)
                except (OSError, ValueError, RuntimeError, TimeoutError, httpx.HTTPError) as exc:
                    with self.lock:
                        self.failed_until = time.monotonic() + 30
                    reason = f"embedding_unavailable:{type(exc).__name__}"
            method = "bm25" if reason else "bm25+bge-m3+chromadb"
            return store.rrf([keyword, semantic], top_k=8), method, reason
