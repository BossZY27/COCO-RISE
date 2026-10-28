"""HTTP trust boundaries remain enforced after retiring the legacy application."""
from types import SimpleNamespace
from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from app.core import security as limits
from app.core.settings import Settings
import app.main as main


@pytest.fixture
def client(monkeypatch):
    settings=Settings(_env_file=None,database_url='postgresql://unused',admin_token='test-token')
    calls=[]
    response={"request_id":str(uuid4()),"status":"clarification_required","answer":"กรุณาระบุชื่อทุน",
              "answer_source":"template","citations":[],"verification":{"ok":True,"problems":[]},"timings_ms":{},
              "processing":{"analysis_method":"rules","retrieval_method":"not_needed","corpus_revision":1}}
    def query(*args): calls.append(args); return response
    runtime=SimpleNamespace(query=SimpleNamespace(run=query),settings=settings,close=lambda:None,
                            status=lambda:{"status":"ok"},database=SimpleNamespace(ready=lambda:True),
                            repository=SimpleNamespace(add_feedback=lambda body:{"id":str(uuid4())}))
    monkeypatch.setattr(main,'Settings',lambda:settings)
    monkeypatch.setattr(main,'Services',lambda _:runtime)
    limits._client_seen.clear(); limits._feedback_client_seen.clear()
    with TestClient(main.app) as c:
        c.query_calls=calls
        yield c


def test_input_limits_and_model_persona_fields_are_rejected(client):
    for body in ({"question":" "},{"question":"a"*401},{"question":"x","history":["a"]*9},
                 {"question":"x","history":["a"*401]}, {"question":"x","model":"unknown"},
                 {"question":"x","persona_id":"P1"}):
        assert client.post('/api/query',json=body).status_code==422
    assert not client.query_calls


def test_context_passes_only_in_its_own_request(client):
    context={"fund":"ทุนวิจัยสถาบัน","fiscal_year":2569,"intent":"amount"}
    assert client.post('/api/query',json={"question":"แล้วทุนนี้ล่ะ","context":context}).status_code==200
    assert client.post('/api/query',json={"question":"แล้วทุนนี้ล่ะ"}).status_code==200
    assert client.query_calls[0][2]['fund']=='ทุนวิจัยสถาบัน'
    assert client.query_calls[1][2] is None


def test_public_security_headers_and_private_cache_policy(client):
    for path in ('/','/admin','/db','/api/status'):
        response=client.get(path)
        assert response.status_code==200
        assert response.headers['x-content-type-options']=='nosniff'
        assert response.headers['x-frame-options']=='DENY'
        assert "frame-ancestors 'none'" in response.headers['content-security-policy']
        if path!='/': assert response.headers['cache-control']=='no-store, private'
    assert client.get('/',headers={'Host':'evil.example'}).status_code==400


@pytest.mark.parametrize('path',['/api/documents','/api/documents/inspect-pdf','/api/facts','/api/rules'])
def test_admin_auth_precedes_operation(client,path):
    assert client.post(path,json={}).status_code==401


def test_body_limit_precedes_json_parse_and_pdf_processing(client):
    assert client.post('/api/query',content=b' '*65537).status_code==413
    assert client.post('/api/documents/inspect-pdf',headers={'Content-Length':str(16*1024*1024)}).status_code==413
    assert client.post('/api/documents',headers={'Content-Length':str(16*1024*1024)}).status_code==413


def test_feedback_is_typed_and_has_separate_rate_limit(client,monkeypatch):
    body={'request_id':str(uuid4()),'helpful':True}
    assert client.post('/api/feedback',json={**body,'note':'a'*2001}).status_code==422
    monkeypatch.setattr(limits,'FEEDBACK_CLIENT_RATE_LIMIT',1)
    assert client.post('/api/feedback',json=body).status_code==201
    assert client.post('/api/feedback',json=body).status_code==429
    assert client.post('/api/query',json={'question':'x'}).status_code==200


def test_rate_limit_cannot_be_changed_by_untrusted_identity_headers(client,monkeypatch):
    monkeypatch.setattr(limits,'CLIENT_RATE_LIMIT',1)
    assert client.post('/api/query',json={'question':'x'},headers={'X-User-Id':'one'}).status_code==200
    response=client.post('/api/query',json={'question':'x'},headers={'X-User-Id':'two'})
    assert response.status_code==429 and 'retry-after' in response.headers
