from types import SimpleNamespace
import httpx
from scripts.model_smoke import run_smoke


def test_timeout_keeps_successful_embedding_in_report():
    def candidate(*args): raise httpx.ReadTimeout('slow model')
    client=SimpleNamespace(settings=SimpleNamespace(embed_model='bge-m3',answer_model='instruct'),
                           embed=lambda *args,**kwargs:[[1.0]*1024],candidate=candidate)
    result=run_smoke(client)
    assert result['embedding']['completed'] and result['embedding']['dimensions']==1024
    assert not result['ok'] and result['answer']['error']=='ReadTimeout'


def test_failed_embedding_does_not_hide_answer_probe():
    def embed(*args,**kwargs): raise httpx.ConnectError('unavailable')
    client=SimpleNamespace(settings=SimpleNamespace(embed_model='bge-m3',answer_model='instruct'),
                           embed=embed,candidate=lambda *args:('คำตอบ','digest'))
    result=run_smoke(client)
    assert not result['ok'] and result['embedding']['error']=='ConnectError'
    assert result['answer']['completed'] and result['answer']['model_revision']=='digest'
