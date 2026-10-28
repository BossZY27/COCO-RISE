"""Real PostgreSQL 17 + pgvector acceptance checks; never use a production database."""
import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text, update
from app.core import security as limits
from app.modules.knowledge import corpus as store
from app.bootstrap import Services
from app.core.errors import ServiceError
from app.core.settings import Settings
from app.infrastructure.postgres.importer import import_legacy
from app.infrastructure.postgres.models import (Base, Chunk, CorpusRevision, Event, Evidence,
    Generation, ImportRun, Job, Knowledge, QueryRun, Version)
from app.infrastructure.postgres.repository import now
from app.modules.knowledge.schemas import Actor, Decision, KnowledgeInput, PageEdit, ReviewDocument
from app.modules.knowledge.context import use_corpus
from app.workers.main import process_one

URL = os.environ.get("RISE_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="Set RISE_TEST_DATABASE_URL to a dedicated database named rise_test")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def migrated():
    assert URL.rsplit("/", 1)[-1] == "rise_test", "Refusing to reset a non-test database"
    env = {**os.environ, "RISE_DATABASE_URL": URL}
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True)


@pytest.fixture
def runtime(migrated, tmp_path):
    settings = Settings(_env_file=None, database_url=URL, files_dir=tmp_path / "pdfs", chroma_dir=tmp_path / "chroma",
                        source_dir=ROOT / "funds-data", admin_token="integration-test-token",
                        llm_base="http://127.0.0.1:1", answer_enabled=False)
    services = Services(settings)
    with services.database.transaction() as s:
        for table in reversed(Base.metadata.sorted_tables):
            if table.name != "corpus_revision":
                s.execute(table.delete())
        s.execute(update(CorpusRevision).values(revision=0))
    import_legacy(services.database, services.files, ROOT / "data", settings.source_dir)
    # Keep tests focused on ingestion rather than attempting a real model download.
    with services.database.transaction() as s:
        s.execute(update(Job).where(Job.kind == "index").values(status="done"))
    yield services
    services.close()


@pytest.fixture
def client(runtime, monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, "Settings", lambda: runtime.settings)
    limits._client_seen.clear()
    limits._feedback_client_seen.clear()
    with TestClient(main.app) as client:
        yield client


def pdf(content="Grant source amount 1250.50 baht"):
    with fitz.open() as doc:
        p = doc.new_page()
        p.insert_text((40, 60), content)
        return doc.tobytes()


def ingest(runtime, content=None):
    result = runtime.repository.create_document(content or pdf(), "source.pdf")
    assert process_one(runtime, uuid.uuid4())
    return result


def publish(runtime, result):
    uid = result["version_id"]
    version = runtime.repository.get_version(uid)
    runtime.repository.review_document(uid, ReviewDocument(reviewer="Human", source_sha256=version["sha256"], confirmed_pages=[1]))
    runtime.repository.publish(uid, "Human")
    return uid


def test_import_is_idempotent_preserves_drafts_and_reviewer_work(runtime):
    with runtime.database.transaction() as s:
        item = s.scalar(select(Knowledge).where(Knowledge.legacy_id == "f01"))
        assert item.status == "draft" and item.reviewed_by is None
        item.payload = {**item.payload, "note": "human correction"}
        rule = s.scalar(select(Knowledge).where(Knowledge.legacy_id == "r04"))
        assert rule.status == "draft" and rule.legacy_document_gate and rule.reviewed_by is None
    again = import_legacy(runtime.database, runtime.files, ROOT / "data", runtime.settings.source_dir)
    assert again["inserted"] == 0
    corpus = runtime.corpora.load()
    assert len(corpus.FACTS) == len(json.loads((ROOT / "data/facts.json").read_text()))
    assert len(corpus.CHUNKS) == 123
    assert next(f for f in corpus.FACTS if f["id"] == "f01")["note"] == "human correction"
    assert not next(f for f in corpus.FACTS if f["id"] == "f01")["verified"]
    assert {r["id"] for r in corpus.RULES} == {"r04"}


def test_parallel_workers_claim_once_and_stale_worker_cannot_commit(runtime):
    result = runtime.repository.create_document(pdf(), "worker.pdf")
    owners = [uuid.uuid4(), uuid.uuid4()]
    with ThreadPoolExecutor(2) as pool:
        claimed = list(pool.map(runtime.repository.claim, owners))
    assert sum(j is not None for j in claimed) == 1
    old = owners[next(i for i, j in enumerate(claimed) if j)]
    with runtime.database.transaction() as s:
        s.execute(update(Job).where(Job.id == result["job_id"]).values(lease_until=now() - timedelta(seconds=1)))
    new = uuid.uuid4()
    assert runtime.repository.claim(new)["id"] == result["job_id"]
    assert not runtime.repository.heartbeat(result["job_id"], old)
    assert not runtime.repository.finish(result["job_id"], old, error="late stale write")
    assert runtime.repository.finish(result["job_id"], new, pages=[{"number":1,"text":"Recovered source", "method":"text","error":None}])


def test_review_publish_supersede_withdraw_and_historical_source(runtime):
    first = ingest(runtime)
    uid = first["version_id"]
    assert not any(c["version_id"] == str(uid) and c["verified"] for c in runtime.corpora.load().CHUNKS)
    with pytest.raises(ServiceError, match="Review"):
        runtime.repository.publish(uid, "Human")
    publish(runtime, first)
    assert any(c["version_id"] == str(uid) and c["verified"] for c in runtime.corpora.load().CHUNKS)
    with pytest.raises(ServiceError, match="new version"):
        runtime.repository.edit_page(uid, 1, PageEdit(text="Changed", reviewer="Human"))
    second = runtime.repository.create_document(pdf("New grant source"), "source-v2.pdf", first["document_id"])
    # Skip any queued index jobs from first publication.
    with runtime.database.transaction() as s:
        s.execute(update(Job).where(Job.kind == "index").values(status="done"))
    process_one(runtime, uuid.uuid4())
    publish(runtime, second)
    assert runtime.repository.get_version(uid)["status"] == "superseded"
    assert runtime.repository.source(uid, page=1)[0].exists()
    runtime.repository.withdraw(second["version_id"], "Human")
    assert not any(c["verified"] and c["document_id"] == str(first["document_id"]) for c in runtime.corpora.load().CHUNKS)
    assert runtime.repository.source(uid, page=1)[0].exists()


def test_knowledge_evidence_transaction_decimal_review_and_no_overwrite(runtime):
    result = ingest(runtime)
    uid = publish(runtime, result)
    chunk = runtime.repository.chunks(uid)[0]
    body = KnowledgeInput(fund="ทุนทดสอบ", amount_type="ceiling", amount="1250.50",
                          evidence=[{"version_id": uid, "page":1, "chunk_id":chunk["id"], "quote":"1250.50"}])
    item = runtime.repository.save_knowledge("fact", body)
    assert str(item["amount"]) == "1250.50"
    assert item["status"] == "draft" and len(item["evidence"]) == 1
    runtime.repository.decide_knowledge("fact", item["id"], Decision(reviewer="Human", decision="approved"))
    with pytest.raises(ServiceError, match="overwritten"):
        runtime.repository.save_knowledge("fact", body, item["id"])
    bad = body.model_copy(update={"evidence": [body.evidence[0].model_copy(update={"quote":"invented"})]})
    with pytest.raises(ServiceError, match="verbatim"):
        runtime.repository.save_knowledge("fact", bad)
    assert len([f for f in runtime.corpora.load().FACTS if f["fund"] == "ทุนทดสอบ"]) == 1


def test_v2_contract_auth_validation_errors_and_feedback(client):
    headers = {"X-Admin-Token":"integration-test-token"}
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/documents", headers={"X-Admin-Token":"bad"}).status_code == 401
    assert client.get("/api/documents?limit=0", headers=headers).status_code == 422
    assert client.get(f"/api/documents/{uuid.uuid4()}", headers=headers).status_code == 404
    assert client.post("/api/query", json={"question":" "}).status_code == 422
    assert client.post("/api/query", json={"question":"x", "extra":True}).status_code == 422
    assert client.post("/api/feedback", json={"request_id":str(uuid.uuid4()),"helpful":True}).status_code == 404
    response = client.post("/api/query", json={"question":"เพดานทุนวิจัยสถาบันเท่าไร"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered" and body["answer_source"] == "template" and body["verification"]["ok"]
    assert body["citations"] and client.get(body["citations"][0]["url"]).status_code == 200
    assert client.post("/api/feedback", json={"request_id":body["request_id"],"helpful":True}).status_code == 201
    assert client.get("/api/ready").json() == {"ready":True}
    assert client.get("/api/status").json()["status"] == "degraded"
    for _ in range(limits.CLIENT_RATE_LIMIT):
        response = client.post("/api/query", json={"question":"x"})
    assert response.status_code == 429 and "retry-after" in response.headers


def test_document_api_complete_flow(client):
    headers = {"X-Admin-Token":"integration-test-token", "Content-Type":"application/pdf"}
    response = client.post("/api/documents", headers=headers, content=pdf())
    assert response.status_code == 202, response.text
    body = response.json()
    assert client.get(f"/api/versions/{body['version_id']}/source", headers=headers).status_code == 200
    assert client.get(f"/api/sources/{body['version_id']}/pages/1").status_code == 404
    assert client.post(f"/api/versions/{body['version_id']}/publish", headers={"X-Admin-Token":"integration-test-token"},
                       json={"reviewer":"Human"}).status_code == 409
    assert client.post("/api/documents", headers=headers, content=b"not a pdf").status_code == 422
    assert client.post("/api/documents", headers=headers, content=pdf()).status_code == 202


def test_withdrawal_during_llm_wait_blocks_result(runtime):
    def withdrawing_candidate(template, citations, timeout=None):
        with runtime.database.transaction() as s:
            v = s.scalar(select(Version).where(Version.current.is_(True)))
            uid = v.id
        runtime.repository.withdraw(uid, "Human")
        return template, "test-digest"
    runtime.settings.answer_enabled = True
    runtime.ollama.candidate = withdrawing_candidate
    result = runtime.query.run("เพดานทุนวิจัยสถาบันเท่าไร", [])
    assert result["status"] == "blocked" and not result["citations"]


def test_only_api_routes_are_served(client):
    for path in ("/v1/query", "/v2/query", "/api/update", "/v1/query/stream"):
        assert client.post(path, json={"question":"มีทุนอะไรบ้าง"}).status_code == 404
    assert client.get("/v1/personas").status_code == 404
    response = client.post("/api/query", json={"question":"เพดานทุนวิจัยสถาบันเท่าไร"})
    assert response.status_code == 200
    assert response.json()["verification"]["ok"]
    assert "persona" not in response.text


def test_pgvector_exact_cosine_generation_is_complete_before_activation(runtime):
    class Embeddings:
        settings = runtime.settings
        def revision(self, model): return "test-model-revision"
        def embed(self, texts, timeout=None): return [[1.0]+[0.0]*1023 for _ in texts]
    rev = runtime.repository.corpus_revision()
    assert not runtime.vectors.ready(rev)
    assert runtime.vectors.build(Embeddings())
    assert runtime.vectors.ready(rev)
    hits = runtime.vectors.search([1.0]+[0.0]*1023, 2569, rev)
    assert hits and all(c["verified"] for c in hits)
    with runtime.database.transaction() as s:
        version_id = s.scalar(select(Version.id).where(Version.current.is_(True)))
    runtime.repository.withdraw(version_id, "Human")
    assert not runtime.vectors.search([1.0]+[0.0]*1023, 2569, rev)


def test_partial_ocr_edit_and_retry_cannot_erase_human_work(runtime):
    result = runtime.repository.create_document(pdf(), "scanned.pdf")
    owner = uuid.uuid4()
    job = runtime.repository.claim(owner)
    runtime.repository.finish(job["id"], owner, pages=[{"number":1,"text":"", "method":"tesseract", "error":"OCR failed"}])
    runtime.repository.edit_page(result["version_id"], 1, PageEdit(text="Human corrected text", reviewer="Human"))
    with pytest.raises(ServiceError, match="human edits"):
        runtime.repository.retry(result["job_id"])
    publish(runtime, result)


def test_import_preserves_quarantine_records_and_human_edit_history(runtime, tmp_path):
    import hashlib
    import shutil
    data = tmp_path / "legacy"
    data.mkdir()
    for name in ("facts.json", "rules.json", "chunks.json", "chunks_ingested.json", "personas.json", "aliases.json"):
        shutil.copy2(ROOT / "data" / name, data / name)
    identifier, job_id = "doc_" + "a"*24, "job_" + "a"*24
    directory = data / "ingestion" / "documents" / identifier
    directory.mkdir(parents=True)
    content = pdf("Human reviewed draft text")
    (directory / "upload.pdf").write_bytes(content)
    (directory / "document.json").write_text(json.dumps({"document_id":identifier,"job_id":job_id,
        "filename":"upload.pdf","content_sha256":hashlib.sha256(content).hexdigest(), "status":"awaiting_review",
        "verified":False,"published":False,"last_edited_at":"2026-10-20T12:00:00"}))
    (directory / "chunks.json").write_text(json.dumps([{"id":"legacy-human-chunk","page":1,
        "text":"Human reviewed draft text","verified":False,"edit_history":[{"reviewer":"Human"}]}]))
    import_legacy(runtime.database, runtime.files, data, runtime.settings.source_dir)
    with runtime.database.transaction() as s:
        job = s.scalar(select(Job).where(Job.legacy_id == job_id))
        chunk = s.scalar(select(Chunk).where(Chunk.legacy_id == "legacy-human-chunk"))
        assert job.status == "awaiting_review" and job.payload["human_edited"]
        assert chunk.text == "Human reviewed draft text" and not chunk.verified
        version = s.get(Version, job.version_id)
        assert version.status == "draft" and version.reviewed_by is None
    assert import_legacy(runtime.database, runtime.files, data, runtime.settings.source_dir)["inserted"] == 0


def test_job_retry_limit_heartbeat_and_index_partial_failure(runtime):
    result = runtime.repository.create_document(pdf(), "retry.pdf")
    for _ in range(runtime.settings.job_max_attempts):
        owner = uuid.uuid4()
        job = runtime.repository.claim(owner)
        assert job["id"] == result["job_id"]
        assert runtime.repository.heartbeat(job["id"], owner)
        runtime.repository.finish(job["id"], owner, error="extractor unavailable")
    assert runtime.repository.get_job(result["job_id"])["status"] == "failed"
    assert runtime.repository.claim(uuid.uuid4()) is None
    runtime.repository.retry(result["job_id"])
    assert runtime.repository.get_job(result["job_id"])["status"] == "queued"
    class BrokenEmbedding:
        settings = runtime.settings
        def revision(self, model): return "broken-model"
        def embed(self, texts, timeout=None): raise ValueError("model failed midway")
    with pytest.raises(ValueError, match="midway"):
        runtime.vectors.build(BrokenEmbedding())
    assert not runtime.vectors.ready(runtime.repository.corpus_revision())


def test_v2_fact_and_rule_api_create_edit_review_contract(client, runtime):
    result = ingest(runtime)
    version_id = publish(runtime, result)
    chunk = runtime.repository.chunks(version_id)[0]
    ref = {"version_id":str(version_id), "page":1, "chunk_id":str(chunk["id"]), "quote":"1250.50"}
    headers = {"X-Admin-Token":"integration-test-token"}
    fact = {"fund":"ทุนทดสอบ", "amount_type":"ceiling", "amount":"1250.50", "evidence":[ref]}
    response = client.post("/api/facts", headers=headers, json=fact)
    assert response.status_code == 201, response.text
    item = response.json()
    assert item["amount"] == "1250.50" and item["status"] == "draft"
    assert client.patch(f"/api/facts/{item['id']}", headers=headers, json=fact).status_code == 200
    assert client.post(f"/api/facts/{item['id']}/review", headers=headers,
                       json={"reviewer":"Human","decision":"approved"}).status_code == 200
    assert client.patch(f"/api/facts/{item['id']}", headers=headers, json=fact).status_code == 409
    rule = {"fund":"ทุนทดสอบ", "conditions":[{"field":"department","op":"eq", "value":"วิทยาลัยการคอมพิวเตอร์",
            "fail_reason":"ผู้ขอรับทุนต้องสังกัดวิทยาลัยการคอมพิวเตอร์"}], "evidence":[ref]}
    response = client.post("/api/rules", headers=headers, json=rule)
    assert response.status_code == 201, response.text
    assert client.post(f"/api/rules/{response.json()['id']}/review", headers=headers,
                       json={"reviewer":"Human","decision":"rejected"}).status_code == 200
    assert client.post("/api/rules", headers=headers, json={**rule,"conditions":[{"field":"password","op":"eq","value":"x","fail_reason":"x"}]}).status_code == 422


def test_feedback_and_query_audit_use_database(client, runtime):
    result = client.post("/api/query", json={"question":"เพดานทุนวิจัยสถาบันเท่าไร"}).json()
    response = client.post("/api/feedback", json={"request_id":result["request_id"], "helpful":True})
    assert response.status_code == 201
    records = client.get("/api/recent", headers={"X-Admin-Token":"integration-test-token"}).json()
    assert any(r["question"] == "เพดานทุนวิจัยสถาบันเท่าไร" for r in records)


def test_document_quota_duplicate_and_safe_pdf_dimensions(runtime):
    runtime.settings.ingestion_max_documents = 5
    content = pdf()
    first = runtime.repository.create_document(content, "quota.pdf")
    duplicate = runtime.repository.create_document(content, "same.pdf")
    assert duplicate["duplicate"] and duplicate["document_id"] == first["document_id"]
    with pytest.raises(ServiceError, match="quota"):
        runtime.repository.create_document(pdf("different"), "over-quota.pdf")
    with fitz.open() as doc:
        doc.new_page(width=20000, height=20000)
        huge_page = doc.tobytes()
    with pytest.raises(ServiceError, match="dimensions"):
        runtime.files.validate(huge_page)
