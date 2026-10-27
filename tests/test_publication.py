"""A new PDF remains quarantined until a named admin confirms every page."""

import hashlib
from contextlib import nullcontext

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from app.modules.knowledge import corpus as store
from app.modules.documents import publication, review
from app.workers import ingestion
from app.main import app


def _pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((40, 60), "Ocean research grant announcement. " * 8)
    value = document.tobytes()
    document.close()
    return value


@pytest.fixture(autouse=True)
def isolated_publication(tmp_path, monkeypatch):
    assert ingestion.wait_for_idle(5)
    monkeypatch.setattr(ingestion, "ROOT", tmp_path / "ingestion")
    monkeypatch.setattr(publication, "publisher", publication.DocumentPublisher(tmp_path / "published.json"))
    monkeypatch.setattr(review, "TOKEN", "test-publisher-token")
    yield
    assert ingestion.wait_for_idle(5)
    store.reload_published_chunks()
    store.refresh_derived_indexes()


def test_admin_must_confirm_source_and_all_pages_before_text_is_searchable(monkeypatch):
    created = ingestion.create_document(_pdf(), "new-grant.pdf")
    monkeypatch.setattr(ingestion, "page_text", lambda _page, _dpi: ("ข้อ 1 Ocean research grant announcement", "text"))
    ingestion.process_job(created["job_id"])
    document_id = created["document_id"]
    document = ingestion.get_document(document_id)
    chunks = ingestion.list_chunks(document_id)
    assert document["status"] == "awaiting_review" and chunks
    chunk_id = chunks[0]["id"]
    assert chunk_id not in {row["id"] for row in store.CHUNKS}
    assert len(store.FACTS) > 0
    fact_count = len(store.FACTS)

    client = TestClient(app)
    path = f"/v1/admin/documents/{document_id}/publish"
    payload = {"reviewer": "เจ้าหน้าที่ผู้ตรวจ", "source_sha256": document["content_sha256"],
               "confirmed_pages": [1]}
    assert client.post(path, json=payload).status_code in (401, 403)
    headers = {"X-Admin-Token": "test-publisher-token"}
    assert client.post(path, json={**payload, "confirmed_pages": []}, headers=headers).status_code == 409
    assert client.post(path, json={**payload, "source_sha256": "0" * 64}, headers=headers).status_code == 409
    assert chunk_id not in {row["id"] for row in store.CHUNKS}

    result = client.post(path, json=payload, headers=headers)
    assert result.status_code == 200, result.text
    assert result.json()["chunk_count"] == len(chunks)
    assert result.json()["already_published"] is False
    assert chunk_id in {row["id"] for row in store.CHUNKS}
    assert [row["id"] for row in store.hierarchical_search("Ocean research grant", None)] == [chunk_id]
    assert chunk_id in {row["id"] for row in store.keyword_search("Ocean research grant", None)}
    assert len(store.FACTS) == fact_count  # scanned monetary hints never become structured facts
    assert client.get(f"/v1/admin/documents/{document_id}/chunks", headers=headers).json()["items"][0]["verified"] is True
    assert client.patch(f"/v1/admin/documents/{document_id}/chunks/{chunk_id}",
                        json={"reviewer": "อีกคน", "text": "แก้หลังเผยแพร่"}, headers=headers).status_code == 400
    assert client.post(path, json=payload, headers=headers).json()["already_published"] is True


def test_incomplete_ocr_cannot_be_published(monkeypatch):
    created = ingestion.create_document(_pdf(), "unreadable.pdf")
    monkeypatch.setattr(ingestion, "page_text", lambda _page, _dpi: ("", "needs_ocr"))
    ingestion.process_job(created["job_id"])
    document = ingestion.get_document(created["document_id"])
    assert document["status"] == "needs_ocr"
    response = TestClient(app).post(
        f"/v1/admin/documents/{created['document_id']}/publish",
        headers={"X-Admin-Token": "test-publisher-token"},
        json={"reviewer": "ผู้ตรวจ", "source_sha256": document["content_sha256"], "confirmed_pages": [1]},
    )
    assert response.status_code == 409


def test_changed_source_pdf_cannot_be_published(monkeypatch):
    created = ingestion.create_document(_pdf(), "tampered.pdf")
    monkeypatch.setattr(ingestion, "page_text", lambda _page, _dpi: ("ข้อ 1 Ocean research grant", "text"))
    ingestion.process_job(created["job_id"])
    document = ingestion.get_document(created["document_id"])
    ingestion.source_path(created["document_id"]).write_bytes(b"not the reviewed PDF")
    response = TestClient(app).post(
        f"/v1/admin/documents/{created['document_id']}/publish",
        headers={"X-Admin-Token": "test-publisher-token"},
        json={"reviewer": "ผู้ตรวจ", "source_sha256": document["content_sha256"], "confirmed_pages": [1]},
    )
    assert response.status_code == 409
    assert all(row.get("document_id") != created["document_id"] for row in store.CHUNKS)


def test_publication_policy_uses_injected_repository_without_publishing_drafts(tmp_path):
    """The approval policy can be tested without PDF parsing or a live index."""
    source = b"reviewed PDF bytes"
    digest = hashlib.sha256(source).hexdigest()
    document_id = "doc_" + "a" * 24

    class MemoryRepository:
        def __init__(self):
            self.document = {"status": "awaiting_review", "pages_total": 2,
                             "needs_ocr_pages": [], "content_sha256": digest}
            self.chunks = [{"id": "chunk_1", "document_id": document_id, "page": 1,
                            "text": "หน้าแรก", "status": "draft", "verified": False, "published": False},
                           {"id": "chunk_2", "document_id": document_id, "page": 2,
                            "text": "หน้าสอง", "status": "draft", "verified": False, "published": False}]
            self.live = []
            self.events = []

        def __bool__(self):
            return False  # dependency injection must not depend on truthiness

        def transaction(self): return nullcontext()
        def valid_document_id(self, value): return value == document_id
        def read_document(self, _value): return self.document
        def read_chunks(self, _value): return self.chunks
        def source_bytes(self, _value): return source
        def read_published(self): return self.live
        def now(self): return "2026-01-01T00:00:00+00:00"

        def write_review(self, _value, chunks, document):
            self.events.append("review")
            self.chunks, self.document = chunks, document

        def publish(self, chunks):
            self.events.append("publish")
            self.live = chunks

    repository = MemoryRepository()
    publisher = publication.DocumentPublisher(tmp_path / "unused.json", repository=repository)
    with pytest.raises(publication.PublicationError, match="ครบทุกหน้า"):
        publisher.publish(document_id, "ผู้ตรวจ", digest, [1])
    assert repository.events == []
    assert all(row["verified"] is False for row in repository.chunks)

    result = publisher.publish(document_id, "ผู้ตรวจ", digest, [1, 2])
    assert result["chunk_count"] == 2
    assert repository.events == ["review", "publish"]
    assert all(row["verified"] is True for row in repository.live)
    assert publisher.publish(document_id, "ผู้ตรวจ", digest, [1, 2])["already_published"] is True
    assert repository.events == ["review", "publish"]
