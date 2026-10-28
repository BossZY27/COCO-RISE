from types import SimpleNamespace
import pytest
from app.modules.knowledge import corpus as store
from app.modules.retrieval.service import HybridRetrieval


def test_real_embedding_path_is_independent_of_answer_mode():
    calls=[]
    embeddings=SimpleNamespace(settings=SimpleNamespace(embed_model='bge-m3',answer_enabled=True),
        revision=lambda m,timeout=1:'same',embed=lambda texts,timeout=None:calls.append(texts) or [[1.0]*1024])
    vectors=SimpleNamespace(ready=lambda r:True,model_revision=lambda r:'same',search=lambda v,y,r:[])
    _,method,reason=HybridRetrieval(vectors,embeddings,2).search('ทุนวิจัย',2569,store.view())
    assert calls==[['ทุนวิจัย']]
    assert method=='bm25+bge-m3+chromadb' and reason is None


def test_unready_index_uses_bm25_without_calling_model():
    embeddings=SimpleNamespace(settings=SimpleNamespace(embed_model='bge-m3'))
    vectors=SimpleNamespace(ready=lambda r:False)
    _,method,reason=HybridRetrieval(vectors,embeddings,2).search('ทุนวิจัย',2569,store.view())
    assert method=='bm25' and reason=='index_not_ready'


def test_index_outage_is_reported_as_fallback():
    embeddings=SimpleNamespace(settings=SimpleNamespace(embed_model='bge-m3'),revision=lambda m,timeout=1:'same',embed=lambda texts,timeout=None:[[1.0]*1024])
    def fail(*args): raise RuntimeError('index offline')
    vectors=SimpleNamespace(ready=lambda r:True,model_revision=lambda r:'same',search=fail)
    retrieval=HybridRetrieval(vectors,embeddings,2)
    _,method,reason=retrieval.search('ทุนวิจัย',2569,store.view())
    assert method=='bm25' and reason=='embedding_unavailable:RuntimeError'
    assert retrieval.search('ทุนวิจัย',2569,store.view())[2]=='embedding_cooldown'


def test_chroma_client_open_error_is_normalized_for_bm25_fallback(monkeypatch):
    from contextlib import contextmanager
    from uuid import uuid4
    from chromadb.errors import InternalError
    from app.infrastructure.chroma import vectors as adapter
    @contextmanager
    def transaction():
        yield SimpleNamespace(scalar=lambda query:uuid4())
    def fail(*args): raise InternalError('unable to open database file')
    monkeypatch.setattr(adapter,'get_chroma_client',fail)
    repository=adapter.ChromaVectorRepository(SimpleNamespace(transaction=transaction))
    with pytest.raises(RuntimeError,match='Vector index unavailable'):
        repository.search([1.0]*1024,2569,1)
