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
