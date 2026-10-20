"""Human-controlled promotion of reviewed PDF passages into live retrieval.

OCR only prepares drafts.  This service requires an authenticated admin action,
an exact source checksum, and explicit acknowledgement of every PDF page.  It
publishes passages, not inferred money facts or eligibility rules.
"""

import hashlib
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

from app.modules.knowledge import corpus as store
from .publication_storage import LocalPublicationRepository


class PublicationError(ValueError):
    """A document cannot safely move from quarantine to the answer corpus."""


class PublicationPort(Protocol):
    """Side effects needed by publication; policy does not know file layout."""

    def transaction(self) -> AbstractContextManager[None]: ...
    def valid_document_id(self, document_id: str) -> bool: ...
    def read_document(self, document_id: str): ...
    def read_chunks(self, document_id: str): ...
    def source_bytes(self, document_id: str) -> bytes | None: ...
    def read_published(self): ...
    def write_review(self, document_id: str, chunks: list[dict], document: dict) -> None: ...
    def publish(self, chunks: list[dict]) -> None: ...
    def now(self) -> str: ...


def _check_document(document: object, confirmed_pages: list[int]) -> dict:
    if not isinstance(document, dict):
        raise PublicationError("ไม่พบเอกสารนำเข้า")
    if document.get("status") != "awaiting_review" or document.get("needs_ocr_pages"):
        raise PublicationError("ต้องอ่าน PDF ทุกหน้าให้ครบก่อน จึงอนุมัติได้")
    page_count = document.get("pages_total")
    if type(page_count) is not int or page_count < 1 or (
        len(confirmed_pages) != page_count or set(confirmed_pages) != set(range(1, page_count + 1))
    ):
        raise PublicationError("ต้องยืนยันว่าตรวจ PDF ครบทุกหน้าก่อนเผยแพร่")
    return document


def _check_chunks(rows: object, document_id: str, confirmed_pages: list[int]) -> set[str]:
    if not isinstance(rows, list) or not rows:
        raise PublicationError("ไม่มีข้อความให้ตรวจอนุมัติ")
    identifiers: set[str] = set()
    for row in rows:
        if (not isinstance(row, dict) or row.get("document_id") != document_id
                or not isinstance(row.get("id"), str) or row["id"] in identifiers
                or not isinstance(row.get("text"), str) or not row["text"].strip()
                or type(row.get("page")) is not int or row["page"] not in confirmed_pages):
            raise PublicationError("ข้อมูลข้อความในเอกสารไม่ครบหรือไม่ถูกต้อง")
        identifiers.add(row["id"])
    return identifiers


def _check_published(existing: object) -> list[dict]:
    if existing is None:
        existing = []
    if (not isinstance(existing, list) or any(
        not isinstance(row, dict) or row.get("status") != "active"
        or row.get("verified") is not True or row.get("published") is not True
        for row in existing
    )):
        raise PublicationError("คลังข้อความที่เผยแพร่ไม่ถูกต้อง")
    return existing


def _already_published(existing: list[dict], identifiers: set[str], document_id: str) -> bool:
    published_ids = {row.get("id") for row in existing}
    if identifiers & published_ids:
        if identifiers <= published_ids and all(
            row.get("document_id") == document_id
            for row in existing if row.get("id") in identifiers
        ):
            return True
        raise PublicationError("พบข้อมูลที่เผยแพร่ซ้ำเพียงบางส่วน ต้องตรวจสถานะก่อน")
    return False


class DocumentPublisher:
    def __init__(self, published_path: Path, repository: PublicationPort | None = None):
        self.repository = repository if repository is not None else LocalPublicationRepository(published_path)
        self._lock = threading.RLock()

    def publish(self, document_id: str, reviewer: str, source_sha256: str,
                confirmed_pages: list[int]) -> dict:
        from app.core.runtime import services as runtime_services
        if runtime_services.get() is not None:
            return runtime_services.get().legacy.publish(document_id, reviewer, source_sha256, confirmed_pages)
        reviewer = reviewer.strip() if isinstance(reviewer, str) else ""
        if not reviewer:
            raise PublicationError("กรุณาระบุชื่อผู้ตรวจ")
        if not isinstance(confirmed_pages, list) or any(type(page) is not int for page in confirmed_pages):
            raise PublicationError("กรุณายืนยันหมายเลขหน้าที่ตรวจทุกหน้า")
        if not self.repository.valid_document_id(document_id):
            raise PublicationError("รหัสเอกสารไม่ถูกต้อง")

        with self._lock, self.repository.transaction():
            document = _check_document(self.repository.read_document(document_id), confirmed_pages)
            source = self.repository.source_bytes(document_id)
            digest = document.get("content_sha256")
            if (source is None or not isinstance(digest, str)
                    or source_sha256 != digest or hashlib.sha256(source).hexdigest() != digest):
                raise PublicationError("ไฟล์ PDF ต้นฉบับหรือรหัสตรวจไฟล์ไม่ตรงกัน")
            rows = self.repository.read_chunks(document_id)
            identifiers = _check_chunks(rows, document_id, confirmed_pages)
            existing = _check_published(self.repository.read_published())
            if _already_published(existing, identifiers, document_id):
                return {"document_id": document_id, "published": True,
                        "chunk_count": len(rows), "already_published": True}
            if document.get("published") is True:
                # A previous attempt may have stopped after updating the
                # quarantine mirror but before the final live-file replace.
                if any(row.get("verified") is not True for row in rows):
                    raise PublicationError("สถานะอนุมัติของเอกสารไม่สอดคล้องกัน")
                approved = rows
            else:
                draft_rows = all(row.get("status") == "draft" and row.get("verified") is False
                                 and row.get("published") is False for row in rows)
                resumed_rows = all(row.get("status") == "active" and row.get("verified") is True
                                   and row.get("published") is True for row in rows)
                if not draft_rows and not resumed_rows:
                    raise PublicationError("พบข้อความที่ไม่ได้อยู่ในสถานะรอตรวจ")
                now = self.repository.now()
                approved = ([{**row, "status": "active", "verified": True, "published": True,
                              "reviewed_by": reviewer, "reviewed_at": now, "updated_at": now}
                             for row in rows] if draft_rows else rows)
                updated_document = {**document, "verified": True, "published": True,
                                    "reviewed_by": reviewer, "reviewed_at": now, "updated_at": now}
                self.repository.write_review(document_id, approved, updated_document)

            # This is the commit point: before it, even a crash leaves the
            # document outside live retrieval.  The operation is retryable.
            self.repository.publish(existing + approved)
            return {"document_id": document_id, "published": True,
                    "chunk_count": len(approved), "already_published": False}


publisher = DocumentPublisher(store.DATA / "chunks_published.json")
