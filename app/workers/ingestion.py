"""Persistent, quarantined ingestion for administrator-uploaded PDFs.

This module deliberately stops at *draft chunks*.  It never writes to the live
retrieval corpus, facts, rules, or embeddings.  The separate publication
service can consume these records only after an explicit human review action.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

import pymupdf as fitz

from scripts.ingest import doc_label, page_text

from app.modules.documents.chunking import section_label_for, split_structured_page


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def _configured_root() -> Path:
    configured = os.getenv("RISE_INGESTION_ROOT") or os.getenv("RISE_INGESTION_DIR")
    return Path(configured or (_PROJECT_ROOT / "data" / "ingestion")).expanduser()


ROOT = _configured_root()

MAX_BYTES = 15 * 1024 * 1024
MAX_PAGES = 50
MAX_PAGE_POINTS = 1800
OCR_DPI = 200
MAX_CHARS = 900
DEFAULT_MAX_QUARANTINE_DOCUMENTS = 100
DEFAULT_MAX_QUARANTINE_BYTES = 512 * 1024 * 1024

_DOCUMENT_ID = re.compile(r"^doc_[0-9a-f]{24}$")
_JOB_ID = re.compile(r"^job_[0-9a-f]{24}$")
_SECTION_TYPE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_AMOUNT = re.compile(r"(?<!\d)\d[\d,]*(?:\.\d+)?\s*บาท")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_FORBIDDEN = re.compile(r'[<>:"/\\|?*]')
_WINDOWS_RESERVED = re.compile(r"^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)

_LOCK = threading.RLock()
_OCR_WORK_LOCK = threading.Lock()
_ACTIVE_JOBS: set[str] = set()
_JOB_QUEUE: queue.Queue[str] = queue.Queue()
_WORKER_CONDITION = threading.Condition()
_WORKER_THREAD: threading.Thread | None = None
_SCHEDULED_JOBS: set[str] = set()
_RERUN_REQUESTED: set[str] = set()
_FINAL_STATUSES = {"awaiting_review", "needs_ocr", "failed"}
_INCOMPLETE_STATUSES = {"queued", "extracting", "chunking"}
_RETRYABLE_STATUSES = {"needs_ocr", "failed"}
_EDITABLE_FIELDS = {"text", "clause", "section_type", "section_label"}
_JOB_DOCUMENT_FIELDS = {
    "status",
    "pages_total",
    "pages_processed",
    "chunk_count",
    "error",
    "processed_at",
    "needs_ocr_pages",
}
_T = TypeVar("_T")


class IngestionStorageError(RuntimeError):
    """A persisted ingestion record exists but cannot be read safely."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_root() -> None:
    (ROOT / "documents").mkdir(parents=True, exist_ok=True)
    (ROOT / "jobs").mkdir(parents=True, exist_ok=True)


def _positive_env_integer(name: str, default: int) -> int:
    """Read a positive upload-quota setting and fail closed on bad config."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"ค่า {name} ต้องเป็นจำนวนเต็มบวก") from exc
    if value <= 0:
        raise ValueError(f"ค่า {name} ต้องเป็นจำนวนเต็มบวก")
    return value


def _quarantine_usage_locked() -> tuple[int, int]:
    """Return document count and stored-PDF bytes while ``_LOCK`` is held.

    Counting directories also accounts for an upload interrupted after its PDF
    was written but before metadata was committed.  Byte accounting uses the
    actual source files rather than trusting mutable JSON metadata.
    """
    documents = ROOT / "documents"
    if not documents.is_dir():
        return 0, 0
    count = 0
    total_bytes = 0
    try:
        directories = list(documents.iterdir())
        for directory in directories:
            if not directory.is_dir() or _DOCUMENT_ID.fullmatch(directory.name) is None:
                continue
            count += 1
            for source in directory.glob("*.pdf"):
                if source.is_file():
                    total_bytes += source.stat().st_size
    except OSError as exc:
        raise IngestionStorageError("ตรวจสอบพื้นที่กักกันเอกสารไม่สำเร็จ") from exc
    return count, total_bytes


def _check_quarantine_capacity_locked(directory: Path, stored_name: str,
                                      incoming_bytes: int) -> None:
    """Reserve capacity for a new source PDF under the ingestion lock."""
    max_documents = _positive_env_integer(
        "RISE_INGESTION_MAX_DOCUMENTS",
        DEFAULT_MAX_QUARANTINE_DOCUMENTS,
    )
    max_total_bytes = _positive_env_integer(
        "RISE_INGESTION_MAX_TOTAL_BYTES",
        DEFAULT_MAX_QUARANTINE_BYTES,
    )
    document_count, stored_bytes = _quarantine_usage_locked()
    new_document_count = document_count + (0 if directory.is_dir() else 1)
    if new_document_count > max_documents:
        raise ValueError(
            "โควตาจำนวนเอกสารกักกันเต็มแล้ว "
            f"(สูงสุด {max_documents} เอกสาร) กรุณานำเอกสารเก่าออกก่อนอัปโหลดไฟล์ใหม่"
        )

    destination = directory / stored_name
    replaced_bytes = 0
    try:
        if destination.is_file():
            replaced_bytes = destination.stat().st_size
    except OSError as exc:
        raise IngestionStorageError("ตรวจสอบพื้นที่กักกันเอกสารไม่สำเร็จ") from exc
    projected_bytes = stored_bytes - replaced_bytes + incoming_bytes
    if projected_bytes > max_total_bytes:
        max_megabytes = max_total_bytes / (1024 * 1024)
        raise ValueError(
            "โควตาพื้นที่ PDF กักกันรวมเต็มแล้ว "
            f"(สูงสุด {max_megabytes:g} MB) กรุณานำเอกสารเก่าออกก่อนอัปโหลดไฟล์ใหม่"
        )


def _document_dir(document_id: str) -> Path | None:
    if not isinstance(document_id, str) or _DOCUMENT_ID.fullmatch(document_id) is None:
        return None
    return ROOT / "documents" / document_id


def _job_file(job_id: str) -> Path | None:
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        return None
    return ROOT / "jobs" / f"{job_id}.json"


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeError, OSError) as exc:
        raise IngestionStorageError(f"อ่านไฟล์สถานะ {path.name} ไม่ได้") from exc


def _atomic_json(path: Path, value: Any) -> None:
    """Write JSON beside its destination, fsync it, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp = Path(handle.name)
    try:
        with handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def _reconcile_document_locked(directory: Path, document: dict, job: dict) -> dict:
    """Repair a document mirror from its authoritative persisted job record."""
    changed = False
    for key in _JOB_DOCUMENT_FIELDS:
        if key in job and document.get(key) != job[key]:
            document[key] = job[key]
            changed = True
    if changed:
        document["verified"] = False
        document["published"] = False
        document["reconciled_at"] = _now()
        _atomic_json(directory / "document.json", document)
    return document


def _job_from_document(document: dict) -> dict | None:
    """Rebuild the queued job side of an interrupted two-record upload.

    ``create_document`` writes the source and document metadata before the job
    record.  If the process stops between those atomic writes, the document is
    still a safe, quarantined draft and contains enough information to recreate
    the deterministic job during startup recovery.
    """
    job_id = document.get("job_id")
    document_id = document.get("document_id")
    status = document.get("status")
    if (
        not isinstance(job_id, str)
        or _JOB_ID.fullmatch(job_id) is None
        or not isinstance(document_id, str)
        or _DOCUMENT_ID.fullmatch(document_id) is None
        or status not in _INCOMPLETE_STATUSES
    ):
        return None

    def nonnegative_integer(field: str) -> int:
        value = document.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise IngestionStorageError(f"ข้อมูลสถานะเอกสารช่อง {field} ไม่ถูกต้อง")
        return value

    created_at = document.get("created_at")
    updated_at = document.get("updated_at", created_at)
    if not isinstance(created_at, str) or not created_at:
        raise IngestionStorageError("ข้อมูลสถานะเอกสารช่อง created_at ไม่ถูกต้อง")
    if not isinstance(updated_at, str) or not updated_at:
        raise IngestionStorageError("ข้อมูลสถานะเอกสารช่อง updated_at ไม่ถูกต้อง")
    error = document.get("error")
    if error is not None and not isinstance(error, str):
        raise IngestionStorageError("ข้อมูลสถานะเอกสารช่อง error ไม่ถูกต้อง")
    needs_ocr_pages = document.get("needs_ocr_pages", [])
    if (
        not isinstance(needs_ocr_pages, list)
        or any(isinstance(page, bool) or not isinstance(page, int) or page < 1
               for page in needs_ocr_pages)
    ):
        raise IngestionStorageError("ข้อมูลสถานะเอกสารช่อง needs_ocr_pages ไม่ถูกต้อง")

    return {
        "job_id": job_id,
        "document_id": document_id,
        "status": status,
        "attempts": nonnegative_integer("attempts"),
        "pages_total": nonnegative_integer("pages_total"),
        "pages_processed": nonnegative_integer("pages_processed"),
        "chunk_count": nonnegative_integer("chunk_count"),
        "created_at": created_at,
        "updated_at": updated_at,
        "error": error,
        "needs_ocr_pages": list(needs_ocr_pages),
        "recovered_from_document": True,
        "recovered_at": _now(),
    }


def _validate_job_record(job: dict, expected_job_id: str | None = None) -> dict:
    """Reject readable JSON whose persisted job schema is unsafe to execute."""
    job_id = job.get("job_id")
    document_id = job.get("document_id")
    status = job.get("status")
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        raise IngestionStorageError("ข้อมูลสถานะงานช่อง job_id ไม่ถูกต้อง")
    if expected_job_id is not None and job_id != expected_job_id:
        raise IngestionStorageError("รหัสในไฟล์สถานะงานไม่ตรงกับชื่อไฟล์")
    if not isinstance(document_id, str) or _DOCUMENT_ID.fullmatch(document_id) is None:
        raise IngestionStorageError("ข้อมูลสถานะงานช่อง document_id ไม่ถูกต้อง")
    if status not in _INCOMPLETE_STATUSES | _FINAL_STATUSES:
        raise IngestionStorageError("ข้อมูลสถานะงานช่อง status ไม่ถูกต้อง")
    for field in ("attempts", "pages_total", "pages_processed", "chunk_count"):
        value = job.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise IngestionStorageError(f"ข้อมูลสถานะงานช่อง {field} ไม่ถูกต้อง")
    for field in (
        "created_at",
        "updated_at",
        "started_at",
        "processed_at",
        "retry_requested_at",
        "recovered_at",
    ):
        value = job.get(field)
        if field in {"created_at", "updated_at"} and (not isinstance(value, str) or not value):
            raise IngestionStorageError(f"ข้อมูลสถานะงานช่อง {field} ไม่ถูกต้อง")
        if value is not None and not isinstance(value, str):
            raise IngestionStorageError(f"ข้อมูลสถานะงานช่อง {field} ไม่ถูกต้อง")
    error = job.get("error")
    if error is not None and not isinstance(error, str):
        raise IngestionStorageError("ข้อมูลสถานะงานช่อง error ไม่ถูกต้อง")
    needs_ocr_pages = job.get("needs_ocr_pages", [])
    if (
        not isinstance(needs_ocr_pages, list)
        or any(isinstance(page, bool) or not isinstance(page, int) or page < 1
               for page in needs_ocr_pages)
    ):
        raise IngestionStorageError("ข้อมูลสถานะงานช่อง needs_ocr_pages ไม่ถูกต้อง")
    return job


def _read_job_record(path: Path, expected_job_id: str) -> dict | None:
    """Read one job, distinguishing an absent file from a malformed JSON root."""
    value = _read_json(path)
    if value is None:
        if path.is_file():
            raise IngestionStorageError(f"ข้อมูลในไฟล์สถานะงาน {path.name} ไม่ใช่ object")
        return None
    if not isinstance(value, dict):
        raise IngestionStorageError(f"ข้อมูลในไฟล์สถานะงาน {path.name} ไม่ใช่ object")
    return _validate_job_record(value, expected_job_id)


def _safe_filename(filename: str) -> str:
    """Keep a readable basename while treating both slash styles as separators."""
    raw = filename if isinstance(filename, str) else ""
    basename = PurePosixPath(raw.replace("\\", "/")).name
    basename = _CONTROL_CHARS.sub("", basename).strip().strip(".")
    basename = _WINDOWS_FORBIDDEN.sub("_", basename)
    if not basename:
        basename = "document.pdf"
    stem = Path(basename).stem.strip().strip(".") or "document"
    if Path(basename).suffix.lower() != ".pdf":
        raise ValueError("กรุณาเลือกไฟล์ PDF ที่ถูกต้อง")
    # Leave room for the extension and avoid awkward Windows trailing spaces.
    stem = stem[:115].rstrip(" .") or "document"
    if _WINDOWS_RESERVED.fullmatch(stem):
        stem = f"_{stem}"
    return f"{stem}.pdf"


def _validate_pdf(data: bytes) -> int:
    if not data.startswith(b"%PDF-"):
        raise ValueError("กรุณาเลือกไฟล์ PDF ที่ถูกต้อง")
    if len(data) > MAX_BYTES:
        raise ValueError("ไฟล์ PDF ใหญ่เกิน 15 MB")
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError("เปิดไฟล์ PDF ไม่สำเร็จ") from exc
    try:
        if document.needs_pass:
            raise ValueError("ไฟล์ PDF มีรหัสผ่าน กรุณาปลดล็อกก่อนนำเข้า")
        if not 0 < document.page_count <= MAX_PAGES:
            raise ValueError("รองรับไฟล์ PDF ตั้งแต่ 1 ถึง 50 หน้า")
        return document.page_count
    finally:
        document.close()


def create_document(data: bytes, filename: str) -> dict:
    """Save one PDF in quarantine and create a deterministic queued job.

    The content hash is the identity.  Uploading identical bytes under another
    filename returns the original document and job rather than duplicating it.
    """
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.create_document(data, filename)
    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("ข้อมูลไฟล์ PDF ไม่ถูกต้อง")
    payload = bytes(data)
    safe_name = _safe_filename(filename)
    pages_total = _validate_pdf(payload)
    digest = hashlib.sha256(payload).hexdigest()
    document_id = f"doc_{digest[:24]}"
    job_id = f"job_{digest[:24]}"
    created_at = _now()

    with _LOCK:
        _ensure_root()
        directory = ROOT / "documents" / document_id
        document_file = directory / "document.json"
        existing = _read_json(document_file)
        if isinstance(existing, dict) and existing.get("content_sha256") == digest:
            stored_name = existing.get("stored_filename") or existing.get("filename")
            stored_path = directory / str(stored_name)
            if not stored_path.is_file():
                _check_quarantine_capacity_locked(directory, str(stored_name), len(payload))
                _atomic_bytes(stored_path, payload)
            existing_job_id = str(existing.get("job_id") or job_id)
            existing_job_file = _job_file(existing_job_id)
            if existing_job_file is not None and not existing_job_file.is_file():
                _atomic_json(
                    existing_job_file,
                    {
                        "job_id": existing_job_id,
                        "document_id": document_id,
                        "status": existing.get("status", "queued"),
                        "attempts": 0,
                        "pages_total": existing.get("pages_total", pages_total),
                        "pages_processed": existing.get("pages_processed", 0),
                        "chunk_count": existing.get("chunk_count", 0),
                        "created_at": existing.get("created_at", created_at),
                        "updated_at": created_at,
                        "error": None,
                        "needs_ocr_pages": existing.get("needs_ocr_pages", []),
                    },
                )
            persisted_job = (
                _read_job_record(existing_job_file, existing_job_id)
                if existing_job_file is not None
                else None
            )
            if isinstance(persisted_job, dict):
                existing = _reconcile_document_locked(directory, existing, persisted_job)
            return {
                "document_id": document_id,
                "job_id": existing_job_id,
                "status": (
                    persisted_job.get("status", "queued")
                    if isinstance(persisted_job, dict)
                    else existing.get("status", "queued")
                ),
                "duplicate": True,
            }

        stored_name = safe_name
        _check_quarantine_capacity_locked(directory, stored_name, len(payload))
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(directory / stored_name, payload)
        document = {
            "document_id": document_id,
            "job_id": job_id,
            "filename": safe_name,
            "stored_filename": stored_name,
            "content_sha256": digest,
            "content_bytes": len(payload),
            "pages_total": pages_total,
            "chunk_count": 0,
            "status": "queued",
            "verified": False,
            "published": False,
            "created_at": created_at,
            "updated_at": created_at,
        }
        job = {
            "job_id": job_id,
            "document_id": document_id,
            "status": "queued",
            "attempts": 0,
            "pages_total": pages_total,
            "pages_processed": 0,
            "chunk_count": 0,
            "created_at": created_at,
            "updated_at": created_at,
            "error": None,
        }
        _atomic_json(document_file, document)
        _atomic_json(ROOT / "jobs" / f"{job_id}.json", job)
        return {
            "document_id": document_id,
            "job_id": job_id,
            "status": "queued",
            "duplicate": False,
        }


def get_document(document_id: str) -> dict | None:
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.get_document(document_id)
    directory = _document_dir(document_id)
    if directory is None:
        return None
    with _LOCK:
        value = _read_json(directory / "document.json")
        if not isinstance(value, dict):
            return None
        job_id = value.get("job_id")
        job_path = _job_file(job_id) if isinstance(job_id, str) else None
        job = _read_job_record(job_path, job_id) if job_path is not None else None
        if isinstance(job, dict):
            value = _reconcile_document_locked(directory, value, job)
        return dict(value)


def get_job(job_id: str) -> dict | None:
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.get_job(job_id)
    path = _job_file(job_id)
    if path is None:
        return None
    with _LOCK:
        value = _read_job_record(path, job_id)
        return dict(value) if isinstance(value, dict) else None


def list_documents(limit: int | None = None, offset: int = 0) -> list[dict]:
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.list_documents(limit, offset)
    with _LOCK:
        directory = ROOT / "documents"
        if not directory.is_dir():
            return []
        rows = []
        for path in directory.glob("doc_*/document.json"):
            row = _read_json(path)
            if isinstance(row, dict):
                job_id = row.get("job_id")
                job_path = _job_file(job_id) if isinstance(job_id, str) else None
                job = _read_job_record(job_path, job_id) if job_path is not None else None
                if isinstance(job, dict):
                    row = _reconcile_document_locked(path.parent, row, job)
                rows.append(row)
        rows.sort(key=lambda row: (str(row.get("created_at", "")), str(row.get("document_id", ""))),
                  reverse=True)
        start = max(0, int(offset))
        if limit is None:
            return [dict(row) for row in rows[start:]]
        size = max(0, int(limit))
        return [dict(row) for row in rows[start:start + size]]


def list_chunks(document_id: str) -> list[dict]:
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.list_chunks(document_id)
    directory = _document_dir(document_id)
    if directory is None:
        return []
    with _LOCK:
        value = _read_json(directory / "chunks.json")
        if not isinstance(value, list):
            return []
        rows = [dict(row) for row in value if isinstance(row, dict)]
        rows.sort(key=lambda row: (int(row.get("global_order", 0)), str(row.get("id", ""))))
        return rows


def source_path(document_id: str) -> Path | None:
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.source_path(document_id)
    directory = _document_dir(document_id)
    if directory is None:
        return None
    with _LOCK:
        document = _read_json(directory / "document.json")
        if not isinstance(document, dict):
            return None
        stored_name = document.get("stored_filename") or document.get("filename")
        if not isinstance(stored_name, str):
            return None
        candidate = (directory / stored_name).resolve()
        try:
            candidate.relative_to(directory.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None


def _write_status(job_id: str, document_id: str, status: str, **updates: Any) -> dict | None:
    """Update job and document state together while the caller holds ``_LOCK``."""
    job_path = _job_file(job_id)
    directory = _document_dir(document_id)
    if job_path is None or directory is None:
        return None
    job = _read_job_record(job_path, job_id)
    document = _read_json(directory / "document.json")
    if not isinstance(job, dict) or not isinstance(document, dict):
        return None
    timestamp = _now()
    job.update({"status": status, "updated_at": timestamp})
    document.update({"status": status, "updated_at": timestamp})
    for key, value in updates.items():
        job[key] = value
        if key in _JOB_DOCUMENT_FIELDS:
            document[key] = value
    # No processing state may implicitly approve or publish a record.
    document["verified"] = False
    document["published"] = False
    _atomic_json(job_path, job)
    _atomic_json(directory / "document.json", document)
    return job


def _amount_candidates(text: str) -> list[str]:
    return list(dict.fromkeys(match.group().strip() for match in _AMOUNT.finditer(text)))[:25]


def run_ocr_serialized(function: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
    """Run one CPU-heavy PDF/OCR operation at a time in this process."""
    with _OCR_WORK_LOCK:
        return function(*args, **kwargs)


def process_job(job_id: str) -> dict | None:
    """Serialize OCR-heavy ingestion jobs so the public demo keeps CPU headroom."""
    return run_ocr_serialized(_process_job, job_id)


def _worker_loop() -> None:
    """Run persisted ingestion jobs one at a time without occupying request threads."""
    while True:
        job_id = _JOB_QUEUE.get()
        try:
            process_job(job_id)
        except Exception:
            # A bad persisted job must not kill the worker and strand later jobs.
            LOGGER.exception("Unhandled ingestion worker error for %s", job_id)
        finally:
            with _WORKER_CONDITION:
                if job_id in _RERUN_REQUESTED:
                    _RERUN_REQUESTED.discard(job_id)
                    _JOB_QUEUE.put(job_id)
                else:
                    _SCHEDULED_JOBS.discard(job_id)
                _JOB_QUEUE.task_done()
                _WORKER_CONDITION.notify_all()


def start_worker() -> bool:
    """Start this process's singleton ingestion worker, returning whether it was new."""
    global _WORKER_THREAD
    with _WORKER_CONDITION:
        if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
            return False
        _WORKER_THREAD = threading.Thread(
            target=_worker_loop,
            name="rise-ingestion-worker",
            daemon=True,
        )
        _WORKER_THREAD.start()
        return True


def enqueue_job(job_id: str, *, after_current: bool = False) -> bool:
    """Schedule one durable job without duplicating queued or active work.

    ``after_current`` is used by the retry route for the narrow interval where
    a worker has persisted a terminal result but has not yet cleared its
    in-memory scheduling marker.
    """
    from app.core.runtime import services as runtime_services
    if runtime_services.get() is not None:
        return True
    job_path = _job_file(job_id)
    if job_path is None or not job_path.is_file():
        return False
    start_worker()
    with _WORKER_CONDITION:
        if job_id in _SCHEDULED_JOBS:
            if after_current and job_id not in _RERUN_REQUESTED:
                _RERUN_REQUESTED.add(job_id)
                return True
            return False
        _SCHEDULED_JOBS.add(job_id)
        _JOB_QUEUE.put(job_id)
        _WORKER_CONDITION.notify_all()
        return True


def wait_for_idle(timeout: float = 5.0) -> bool:
    """Wait until this process's ingestion queue is idle (primarily for tests)."""
    deadline = time.monotonic() + max(0.0, timeout)
    with _WORKER_CONDITION:
        while _SCHEDULED_JOBS:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _WORKER_CONDITION.wait(timeout=remaining)
        return True


def _process_job(job_id: str) -> dict | None:
    """Extract and chunk a queued PDF without ever publishing it.

    Chunk identifiers depend only on document content and structural order, so
    restarting an interrupted job replaces the same records instead of adding
    duplicates.
    """
    job_path = _job_file(job_id)
    if job_path is None:
        return None
    with _LOCK:
        job = _read_job_record(job_path, job_id)
        if not isinstance(job, dict):
            return None
        document_id = job.get("document_id")
        if not isinstance(document_id, str) or _document_dir(document_id) is None:
            return None
        if job.get("status") in _FINAL_STATUSES:
            return dict(job)
        if job_id in _ACTIVE_JOBS:
            return dict(job)
        _ACTIVE_JOBS.add(job_id)
        attempts = int(job.get("attempts", 0)) + 1

    try:
        with _LOCK:
            _write_status(
                job_id,
                document_id,
                "extracting",
                attempts=attempts,
                pages_processed=0,
                chunk_count=0,
                error=None,
                started_at=_now(),
            )
        path = source_path(document_id)
        document_meta = get_document(document_id)
        if path is None or document_meta is None:
            raise FileNotFoundError("ไม่พบ PDF ต้นฉบับในคลังกักกัน")

        extracted: list[dict] = []
        unreadable_pages: list[int] = []
        with fitz.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError("ไฟล์ PDF มีรหัสผ่าน")
            pages_total = pdf.page_count
            for page_number, page in enumerate(pdf, start=1):
                if max(page.rect.width, page.rect.height) > MAX_PAGE_POINTS:
                    text, method = "", "needs_ocr"
                else:
                    text, method = page_text(page, OCR_DPI)
                if method == "needs_ocr" or not text.strip():
                    unreadable_pages.append(page_number)
                else:
                    extracted.extend(
                        split_structured_page(
                            text,
                            page_number=page_number,
                            extraction_method=method,
                            max_chars=MAX_CHARS,
                        )
                    )
                with _LOCK:
                    _write_status(
                        job_id,
                        document_id,
                        "extracting",
                        attempts=attempts,
                        pages_total=pages_total,
                        pages_processed=page_number,
                        chunk_count=0,
                        error=None,
                    )

        directory = _document_dir(document_id)
        if directory is None:
            return None
        with _LOCK:
            _write_status(
                job_id,
                document_id,
                "chunking",
                attempts=attempts,
                pages_total=document_meta.get("pages_total", 0),
                pages_processed=document_meta.get("pages_total", 0),
                chunk_count=0,
                error=None,
            )

        if not extracted:
            with _LOCK:
                _atomic_json(directory / "chunks.json", [])
                result = _write_status(
                    job_id,
                    document_id,
                    "needs_ocr",
                    attempts=attempts,
                    pages_total=document_meta.get("pages_total", 0),
                    pages_processed=document_meta.get("pages_total", 0),
                    chunk_count=0,
                    error="อ่านข้อความได้แต่ไม่พบส่วนที่นำมาตรวจได้",
                    needs_ocr_pages=unreadable_pages,
                    processed_at=_now(),
                )
                return dict(result) if isinstance(result, dict) else None

        label, fiscal_year = doc_label(Path(document_meta["filename"]))
        digest_prefix = str(document_meta["content_sha256"])[:12]
        timestamp = _now()
        chunks = []
        for global_order, row in enumerate(extracted):
            text = " ".join(str(row.get("text", "")).split())
            chunk = {
                **row,
                "id": f"chunk_{digest_prefix}_{global_order:05d}",
                "document_id": document_id,
                "doc": label,
                "fiscal_year": fiscal_year,
                "source_filename": document_meta["filename"],
                "text": text,
                "global_order": global_order,
                "status": "draft",
                "verified": False,
                "published": False,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            if "amount_candidates" not in chunk:
                chunk["amount_candidates"] = _amount_candidates(text)
            chunks.append(chunk)

        with _LOCK:
            _atomic_json(directory / "chunks.json", chunks)
            final_status = "needs_ocr" if unreadable_pages else "awaiting_review"
            result = _write_status(
                job_id,
                document_id,
                final_status,
                attempts=attempts,
                pages_total=document_meta.get("pages_total", 0),
                pages_processed=document_meta.get("pages_total", 0),
                chunk_count=len(chunks),
                error=None,
                needs_ocr_pages=unreadable_pages,
                processed_at=_now(),
            )
            return dict(result) if isinstance(result, dict) else None
    except Exception as exc:
        with _LOCK:
            directory = _document_dir(str(job.get("document_id", "")))
            existing_chunks = _read_json(directory / "chunks.json") if directory is not None else None
            existing_count = len(existing_chunks) if isinstance(existing_chunks, list) else 0
            result = _write_status(
                job_id,
                str(job.get("document_id", "")),
                "failed",
                attempts=int(job.get("attempts", 0)) + 1,
                chunk_count=existing_count,
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
                processed_at=_now(),
            )
            return dict(result) if isinstance(result, dict) else None
    finally:
        with _LOCK:
            _ACTIVE_JOBS.discard(job_id)


def incomplete_job_ids() -> list[str]:
    """Return persisted non-terminal jobs that should resume after a restart."""
    with _LOCK:
        _ensure_root()
        documents_directory = ROOT / "documents"
        for document_path in documents_directory.glob("doc_*/document.json"):
            document = _read_json(document_path)
            if not isinstance(document, dict):
                continue
            rebuilt_job = _job_from_document(document)
            if rebuilt_job is None:
                continue
            rebuilt_path = _job_file(rebuilt_job["job_id"])
            if rebuilt_path is not None and not rebuilt_path.is_file():
                _atomic_json(rebuilt_path, rebuilt_job)

        directory = ROOT / "jobs"
        rows: list[dict] = []
        for path in directory.glob("job_*.json"):
            row = _read_job_record(path, path.stem)
            if not isinstance(row, dict):
                continue
            document_id = row.get("document_id")
            document_directory = _document_dir(document_id) if isinstance(document_id, str) else None
            if document_directory is not None:
                document = _read_json(document_directory / "document.json")
                if isinstance(document, dict):
                    _reconcile_document_locked(document_directory, document, row)
            if row.get("status") in _INCOMPLETE_STATUSES:
                rows.append(row)
        rows.sort(key=lambda row: (str(row.get("created_at", "")), str(row.get("job_id", ""))))
        return [str(row["job_id"]) for row in rows if _JOB_ID.fullmatch(str(row.get("job_id", "")))]


def resume_incomplete_jobs() -> list[str]:
    """Resume saved queued/extracting/chunking jobs on the singleton worker."""
    job_ids = incomplete_job_ids()
    start_worker()
    for saved_job_id in job_ids:
        enqueue_job(saved_job_id)
    return job_ids


def retry_job(job_id: str) -> dict | None:
    """Reset a failed/needs-OCR job to queued; processing remains explicit."""
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.retry_job(job_id)
    job_path = _job_file(job_id)
    if job_path is None:
        return None
    with _LOCK:
        job = _read_job_record(job_path, job_id)
        if not isinstance(job, dict):
            return None
        status = str(job.get("status", ""))
        if status not in _RETRYABLE_STATUSES:
            raise ValueError("ลองประมวลผลซ้ำได้เฉพาะงานที่ล้มเหลวหรือยังต้อง OCR")
        document_id = str(job.get("document_id", ""))
        directory = _document_dir(document_id)
        if directory is None:
            return None
        chunks = _read_json(directory / "chunks.json")
        chunk_count = len(chunks) if isinstance(chunks, list) else 0
        result = _write_status(
            job_id,
            document_id,
            "queued",
            pages_processed=0,
            chunk_count=chunk_count,
            error=None,
            needs_ocr_pages=[],
            retry_requested_at=_now(),
        )
        return dict(result) if isinstance(result, dict) else None


def update_chunk(
    document_id: str,
    chunk_id: str,
    reviewer: str,
    changes: dict,
) -> dict | None:
    """Record a human correction while keeping the chunk quarantined.

    This function intentionally cannot approve or publish.  Unknown fields are
    rejected rather than silently ignored so a future UI cannot accidentally
    smuggle an approval flag through the correction endpoint.
    """
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.update_chunk(document_id, chunk_id, reviewer, changes)
    reviewer = reviewer.strip() if isinstance(reviewer, str) else ""
    if not reviewer:
        raise ValueError("กรุณาระบุชื่อผู้ตรวจ")
    if not isinstance(changes, dict):
        raise ValueError("ข้อมูลที่ต้องการแก้ไม่ถูกต้อง")
    unknown = set(changes) - _EDITABLE_FIELDS
    if unknown:
        raise ValueError(f"ไม่อนุญาตให้แก้ฟิลด์: {', '.join(sorted(unknown))}")
    if not changes:
        raise ValueError("กรุณาระบุข้อมูลที่ต้องการแก้")

    directory = _document_dir(document_id)
    if directory is None:
        return None
    with _LOCK:
        document = _read_json(directory / "document.json")
        if not isinstance(document, dict):
            return None
        if document.get("published") is True:
            raise ValueError("เอกสารที่เผยแพร่แล้วแก้ข้อความผ่านขั้นตอนฉบับร่างไม่ได้")
        if document.get("status") != "awaiting_review":
            raise ValueError(
                "แก้ข้อความได้เมื่ออ่านทุกหน้าครบและอยู่ในสถานะรอคนตรวจเท่านั้น "
                "เอกสารนี้ยังต้อง OCR หรือประมวลผลใหม่"
            )
        rows = _read_json(directory / "chunks.json")
        if not isinstance(rows, list):
            return None
        target = next(
            (row for row in rows if isinstance(row, dict) and row.get("id") == chunk_id),
            None,
        )
        if target is None:
            return None

        clean: dict[str, str] = {}
        if "text" in changes:
            text = changes["text"]
            if not isinstance(text, str) or not text.strip() or len(text) > 12_000:
                raise ValueError("ข้อความต้องมีความยาว 1 ถึง 12,000 ตัวอักษร")
            clean["text"] = " ".join(text.split())
        if "clause" in changes:
            clause = changes["clause"]
            if not isinstance(clause, str) or len(clause) > 120:
                raise ValueError("ชื่อข้อยาวเกิน 120 ตัวอักษร")
            clean["clause"] = clause.strip()
        if "section_type" in changes:
            section_type = changes["section_type"]
            if not isinstance(section_type, str) or _SECTION_TYPE.fullmatch(section_type) is None:
                raise ValueError("ประเภทส่วนเอกสารไม่ถูกต้อง")
            try:
                canonical_label = section_label_for(section_type)
            except ValueError as exc:
                raise ValueError("ประเภทส่วนเอกสารไม่ถูกต้อง") from exc
            clean["section_type"] = section_type
            if "section_label" not in changes:
                clean["section_label"] = canonical_label
        if "section_label" in changes:
            section_label = changes["section_label"]
            if not isinstance(section_label, str) or len(section_label) > 160:
                raise ValueError("ชื่อส่วนเอกสารยาวเกิน 160 ตัวอักษร")
            clean["section_label"] = section_label.strip()

        edited_at = _now()
        target.update(clean)
        if "text" in clean:
            target["amount_candidates"] = _amount_candidates(clean["text"])
        target.update(
            {
                "status": "draft",
                "verified": False,
                "published": False,
                "last_edited_by": reviewer,
                "last_edited_at": edited_at,
                "updated_at": edited_at,
            }
        )
        history = target.setdefault("edit_history", [])
        if isinstance(history, list):
            history.append(
                {
                    "reviewer": reviewer,
                    "edited_at": edited_at,
                    "fields": sorted(clean),
                }
            )
        _atomic_json(directory / "chunks.json", rows)

        document = _read_json(directory / "document.json")
        if isinstance(document, dict):
            document.update(
                {
                    "updated_at": edited_at,
                    "last_edited_by": reviewer,
                    "last_edited_at": edited_at,
                    "verified": False,
                    "published": False,
                }
            )
            _atomic_json(directory / "document.json", document)
        return dict(target)
