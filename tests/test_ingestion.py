"""Tests for the quarantined PDF-ingestion workflow.

Nothing in this suite may write to the live ``data/`` directory or publish a
chunk into the answer path.  Uploads are draft material until a named human
reviews them.
"""
from pathlib import Path
import sys
import threading
import time

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.modules.knowledge import corpus as store
from app.infrastructure.ollama import legacy as llm
from app.modules.documents import review
from app.modules.documents.chunking import classify_section, split_structured_page
from app.workers import ingestion
from app.main import app


ADMIN_HEADERS = {"X-Admin-Token": "test-admin-token"}


def _sample_pdf(text: str = "Grant announcement text. " * 12) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((40, 60), text)
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture(autouse=True)
def isolated_ingestion(tmp_path, monkeypatch):
    if hasattr(ingestion, "wait_for_idle"):
        assert ingestion.wait_for_idle(5)
    monkeypatch.setattr(ingestion, "ROOT", tmp_path / "ingestion")
    monkeypatch.setattr(review, "TOKEN", "test-admin-token")
    yield
    if hasattr(ingestion, "wait_for_idle"):
        assert ingestion.wait_for_idle(5)


def _wait_for_terminal(job_id: str, timeout: float = 5) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = ingestion.get_job(job_id)
        if job and job["status"] in {"awaiting_review", "needs_ocr", "failed"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach a terminal state")


def test_structure_aware_chunking_hard_caps_a_single_long_ocr_line():
    text = "ข้อ 1 " + ("เนื้อหาทุนวิจัย" * 190) + " ข้อ 2 คุณสมบัติผู้ขอรับทุน"

    chunks = split_structured_page(text, page_number=4, extraction_method="ocr", max_chars=900)

    assert len(chunks) >= 3
    assert all(0 < len(chunk["text"]) <= 900 for chunk in chunks)
    assert [chunk["order_in_page"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk["page"] == 4 and chunk["extraction_method"] == "ocr" for chunk in chunks)
    assert chunks[-1]["clause"] == "ข้อ 2"


def test_decimal_subclause_links_to_the_actual_thai_parent_label():
    chunks = split_structured_page(
        "ข้อ 1 หลักเกณฑ์ทั่วไป 1.1 รายละเอียดข้อย่อย",
        page_number=1,
        extraction_method="text",
    )

    child = next(chunk for chunk in chunks if chunk["clause"] == "1.1")
    assert child["parent_clause"] == "ข้อ 1"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("เงินสมทบ หมายความว่า เงินที่หน่วยงานสนับสนุน", "definition"),
        ("วงเงินสูงสุดต่อทุน 100,000 บาท", "amount_table"),
        ("ผู้ขอรับทุนต้องเป็นบุคลากรในสังกัด", "eligibility"),
        ("ยื่นแบบคำขอพร้อมเอกสารประกอบ", "application"),
    ],
)
def test_section_classifier_is_conservative_and_deterministic(text, expected):
    assert classify_section("ข้อ 1", text)["section_type"] == expected


def test_create_document_persists_pdf_safely_and_deduplicates_by_content():
    payload = _sample_pdf()

    first = ingestion.create_document(payload, "../../ประกาศใหม่.pdf")
    second = ingestion.create_document(payload, "ชื่อไฟล์อีกชื่อ.pdf")

    assert first["document_id"] == second["document_id"]
    assert first["duplicate"] is False and second["duplicate"] is True
    assert ingestion.source_path(first["document_id"]).read_bytes() == payload
    assert ingestion.get_document(first["document_id"])["filename"] == "ประกาศใหม่.pdf"
    assert ingestion.get_job(first["job_id"])["status"] == "queued"


def test_document_quota_rejects_new_content_but_still_allows_duplicate(monkeypatch):
    monkeypatch.setenv("RISE_INGESTION_MAX_DOCUMENTS", "1")
    first_payload = _sample_pdf("First quarantined document. " * 12)
    second_payload = _sample_pdf("Second quarantined document. " * 12)

    first = ingestion.create_document(first_payload, "first.pdf")
    duplicate = ingestion.create_document(first_payload, "same-content.pdf")

    assert duplicate["document_id"] == first["document_id"]
    assert duplicate["duplicate"] is True
    with pytest.raises(ValueError, match="โควตาจำนวนเอกสารกักกัน.*1"):
        ingestion.create_document(second_payload, "second.pdf")
    assert len(ingestion.list_documents()) == 1


def test_total_quarantine_byte_quota_rejects_new_content_but_not_duplicate(monkeypatch):
    first_payload = _sample_pdf("Stored within the byte quota. " * 12)
    second_payload = _sample_pdf("Would exceed the byte quota. " * 12)
    monkeypatch.setenv(
        "RISE_INGESTION_MAX_TOTAL_BYTES",
        str(len(first_payload) + len(second_payload) - 1),
    )

    first = ingestion.create_document(first_payload, "first.pdf")
    duplicate = ingestion.create_document(first_payload, "duplicate.pdf")

    assert duplicate["document_id"] == first["document_id"]
    assert duplicate["duplicate"] is True
    with pytest.raises(ValueError, match="โควตาพื้นที่ PDF กักกันรวม"):
        ingestion.create_document(second_payload, "second.pdf")
    assert ingestion.source_path(first["document_id"]).read_bytes() == first_payload


def test_duplicate_with_missing_source_must_fit_byte_quota_before_restore(monkeypatch):
    payload = _sample_pdf("Source PDF must be restored within quota. " * 12)
    created = ingestion.create_document(payload, "original.pdf")
    directory = ingestion._document_dir(created["document_id"])
    source = directory / ingestion.get_document(created["document_id"])["stored_filename"]
    source.unlink()
    monkeypatch.setenv("RISE_INGESTION_MAX_TOTAL_BYTES", str(len(payload) - 1))

    with pytest.raises(ValueError, match="โควตาพื้นที่ PDF กักกันรวม"):
        ingestion.create_document(payload, "duplicate.pdf")

    assert not source.exists()


@pytest.mark.parametrize(
    "name,value",
    [
        ("RISE_INGESTION_MAX_DOCUMENTS", "not-a-number"),
        ("RISE_INGESTION_MAX_TOTAL_BYTES", "0"),
    ],
)
def test_invalid_quarantine_quota_configuration_fails_closed(monkeypatch, name, value):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        ingestion.create_document(_sample_pdf(), "new.pdf")

    assert ingestion.list_documents() == []


def test_document_quota_is_checked_atomically_for_concurrent_uploads(monkeypatch):
    monkeypatch.setenv("RISE_INGESTION_MAX_DOCUMENTS", "1")
    gate = threading.Barrier(3)
    results: list[dict] = []
    errors: list[Exception] = []

    def upload(payload: bytes, filename: str) -> None:
        gate.wait()
        try:
            results.append(ingestion.create_document(payload, filename))
        except Exception as exc:  # Captured for assertions in the test thread.
            errors.append(exc)

    threads = [
        threading.Thread(
            target=upload,
            args=(_sample_pdf(f"Concurrent document {index}. " * 12), f"{index}.pdf"),
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    gate.wait()
    for thread in threads:
        thread.join(5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "โควตาจำนวนเอกสารกักกัน" in str(errors[0])
    assert len(ingestion.list_documents()) == 1


def test_blank_ingestion_env_uses_the_safe_default(monkeypatch):
    monkeypatch.setenv("RISE_INGESTION_ROOT", "")
    monkeypatch.setenv("RISE_INGESTION_DIR", "")

    assert ingestion._configured_root() == Path(ingestion.__file__).resolve().parents[2] / "data" / "ingestion"


def test_processing_creates_ordered_draft_chunks_and_is_idempotent(monkeypatch):
    created = ingestion.create_document(_sample_pdf(), "ประกาศ_77-69.pdf")
    monkeypatch.setattr(
        ingestion,
        "page_text",
        lambda _page, _dpi: (
            "ข้อ 1 เงินสมทบ หมายความว่า เงินสนับสนุน\n"
            "ข้อ 2 ผู้ขอรับทุนต้องเป็นบุคลากรในสังกัด\n"
            "ข้อ 3 วงเงินสูงสุด 100,000 บาท",
            "ocr",
        ),
    )

    ingestion.process_job(created["job_id"])
    first_ids = [row["id"] for row in ingestion.list_chunks(created["document_id"])]
    ingestion.process_job(created["job_id"])
    chunks = ingestion.list_chunks(created["document_id"])

    assert ingestion.get_job(created["job_id"])["status"] == "awaiting_review"
    assert ingestion.get_document(created["document_id"])["status"] == "awaiting_review"
    assert [row["id"] for row in chunks] == first_ids
    assert [row["global_order"] for row in chunks] == list(range(len(chunks)))
    assert all(row["status"] == "draft" and row["verified"] is False for row in chunks)
    assert all(row["published"] is False and row["extraction_method"] == "ocr" for row in chunks)
    assert chunks[-1]["amount_candidates"] == ["100,000 บาท"]


def test_unreadable_scan_stays_out_of_reviewable_chunks(monkeypatch):
    created = ingestion.create_document(_sample_pdf(), "scan.pdf")
    monkeypatch.setattr(ingestion, "page_text", lambda _page, _dpi: ("", "needs_ocr"))

    ingestion.process_job(created["job_id"])

    assert ingestion.get_job(created["job_id"])["status"] == "needs_ocr"
    assert ingestion.list_chunks(created["document_id"]) == []


def test_partly_unreadable_pdf_keeps_readable_drafts_and_flags_missing_pages(monkeypatch):
    document = fitz.open()
    document.new_page().insert_text((40, 60), "Readable page")
    document.new_page().insert_text((40, 60), "Scanned page")
    payload = document.tobytes()
    document.close()
    created = ingestion.create_document(payload, "mixed.pdf")

    def extract(page, _dpi):
        if page.number == 0:
            return "ข้อ 1 ข้อมูลที่อ่านได้", "text"
        return "", "needs_ocr"

    monkeypatch.setattr(ingestion, "page_text", extract)
    ingestion.process_job(created["job_id"])

    job = ingestion.get_job(created["job_id"])
    saved_document = ingestion.get_document(created["document_id"])
    chunks = ingestion.list_chunks(created["document_id"])
    assert job["status"] == "needs_ocr"
    assert job["needs_ocr_pages"] == [2]
    assert job["chunk_count"] == 1
    assert saved_document["needs_ocr_pages"] == [2]
    assert len(chunks) == 1
    assert chunks[0]["page"] == 1
    assert chunks[0]["status"] == "draft"
    assert chunks[0]["verified"] is False and chunks[0]["published"] is False

    with pytest.raises(ValueError, match="OCR"):
        ingestion.update_chunk(
            created["document_id"],
            chunks[0]["id"],
            "ผู้ตรวจ",
            {"text": "ข้อความที่แก้แล้ว"},
        )


def test_retry_failure_preserves_readable_drafts(monkeypatch):
    document = fitz.open()
    document.new_page().insert_text((40, 60), "Readable page")
    document.new_page().insert_text((40, 60), "Scanned page")
    payload = document.tobytes()
    document.close()
    created = ingestion.create_document(payload, "mixed-retry.pdf")
    monkeypatch.setattr(
        ingestion,
        "page_text",
        lambda page, _dpi: ("ข้อ 1 ข้อมูลที่อ่านได้", "text") if page.number == 0 else ("", "needs_ocr"),
    )
    ingestion.process_job(created["job_id"])
    before = ingestion.list_chunks(created["document_id"])

    ingestion.retry_job(created["job_id"])
    monkeypatch.setattr(
        ingestion,
        "page_text",
        lambda _page, _dpi: (_ for _ in ()).throw(RuntimeError("temporary OCR failure")),
    )
    ingestion.process_job(created["job_id"])

    assert ingestion.get_job(created["job_id"])["status"] == "failed"
    assert ingestion.list_chunks(created["document_id"]) == before


def test_document_status_reconciles_from_the_authoritative_job(monkeypatch):
    created = ingestion.create_document(_sample_pdf(), "reconcile.pdf")
    monkeypatch.setattr(ingestion, "page_text", lambda _page, _dpi: ("ข้อ 1 ข้อมูล", "text"))
    ingestion.process_job(created["job_id"])
    directory = ingestion._document_dir(created["document_id"])
    document = ingestion.get_document(created["document_id"])
    document["status"] = "chunking"
    ingestion._atomic_json(directory / "document.json", document)

    repaired = ingestion.get_document(created["document_id"])
    assert repaired["status"] == "awaiting_review"
    assert repaired["chunk_count"] == 1


def test_incomplete_jobs_are_discoverable_for_restart_recovery(monkeypatch):
    queued = ingestion.create_document(_sample_pdf(), "queued.pdf")
    failed = ingestion.create_document(_sample_pdf("Different content. " * 12), "failed.pdf")
    monkeypatch.setattr(ingestion, "page_text", lambda _page, _dpi: (_ for _ in ()).throw(RuntimeError("boom")))
    ingestion.process_job(failed["job_id"])

    assert ingestion.incomplete_job_ids() == [queued["job_id"]]


def test_worker_queue_is_fifo_serial_and_deduplicates_pending_jobs(monkeypatch):
    created = [
        ingestion.create_document(_sample_pdf(f"Document {index}. " * 12), f"{index}.pdf")
        for index in range(3)
    ]
    calls: list[str] = []
    state = {"active": 0, "maximum": 0}
    lock = threading.Lock()

    def fake_process(job_id: str):
        with lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
            calls.append(job_id)
        time.sleep(0.02)
        with lock:
            state["active"] -= 1

    monkeypatch.setattr(ingestion, "process_job", fake_process)

    assert ingestion.enqueue_job(created[0]["job_id"]) is True
    assert ingestion.enqueue_job(created[0]["job_id"]) is False
    assert ingestion.enqueue_job(created[1]["job_id"]) is True
    assert ingestion.enqueue_job(created[2]["job_id"]) is True
    assert ingestion.wait_for_idle(5) is True

    assert calls == [row["job_id"] for row in created]
    assert state["maximum"] == 1


def test_initial_status_write_failure_clears_active_job_marker(monkeypatch):
    created = ingestion.create_document(_sample_pdf(), "status-write-failure.pdf")

    def fail_status_write(*_args, **_kwargs):
        raise OSError("simulated status write failure")

    monkeypatch.setattr(ingestion, "_write_status", fail_status_write)
    try:
        with pytest.raises(OSError, match="simulated status write failure"):
            ingestion._process_job(created["job_id"])
        assert created["job_id"] not in ingestion._ACTIVE_JOBS
    finally:
        ingestion._ACTIVE_JOBS.discard(created["job_id"])


def test_retry_requested_during_worker_cleanup_runs_exactly_once_more(monkeypatch):
    created = ingestion.create_document(_sample_pdf(), "retry-race.pdf")
    entered_terminal_window = threading.Event()
    release_first_run = threading.Event()
    calls: list[str] = []

    def fake_process(job_id: str):
        calls.append(job_id)
        if len(calls) == 1:
            with ingestion._LOCK:
                ingestion._write_status(
                    job_id,
                    created["document_id"],
                    "needs_ocr",
                    needs_ocr_pages=[1],
                )
            entered_terminal_window.set()
            assert release_first_run.wait(5)
        else:
            with ingestion._LOCK:
                ingestion._write_status(
                    job_id,
                    created["document_id"],
                    "awaiting_review",
                    needs_ocr_pages=[],
                )

    monkeypatch.setattr(ingestion, "process_job", fake_process)
    assert ingestion.enqueue_job(created["job_id"])
    assert entered_terminal_window.wait(5)
    try:
        ingestion.retry_job(created["job_id"])
        assert ingestion.enqueue_job(created["job_id"], after_current=True)
    finally:
        release_first_run.set()

    assert ingestion.wait_for_idle(5)
    assert calls == [created["job_id"], created["job_id"]]
    assert ingestion.get_job(created["job_id"])["status"] == "awaiting_review"


def test_resume_incomplete_jobs_enqueues_persisted_jobs_in_recovery_order(monkeypatch):
    created = [
        ingestion.create_document(_sample_pdf(f"Recovery {index}. " * 12), f"recovery-{index}.pdf")
        for index in range(2)
    ]
    calls: list[str] = []
    monkeypatch.setattr(ingestion, "process_job", lambda job_id: calls.append(job_id))

    recovered = ingestion.resume_incomplete_jobs()

    assert ingestion.wait_for_idle(5)
    assert set(recovered) == {row["job_id"] for row in created}
    assert calls == recovered


def test_restart_recovery_rebuilds_a_missing_job_from_queued_document():
    created = ingestion.create_document(_sample_pdf(), "interrupted-upload.pdf")
    job_path = ingestion._job_file(created["job_id"])
    job_path.unlink()

    recovered = ingestion.incomplete_job_ids()

    assert recovered == [created["job_id"]]
    rebuilt = ingestion.get_job(created["job_id"])
    assert rebuilt["document_id"] == created["document_id"]
    assert rebuilt["status"] == "queued"
    assert rebuilt["pages_total"] == 1
    assert rebuilt["pages_processed"] == 0
    document = ingestion.get_document(created["document_id"])
    assert document["verified"] is False and document["published"] is False


@pytest.mark.parametrize("payload", [b"{broken", b"\xff"])
def test_corrupt_persisted_job_is_reported_instead_of_silently_skipped(payload):
    jobs = ingestion.ROOT / "jobs"
    jobs.mkdir(parents=True)
    (jobs / f"job_{'a' * 24}.json").write_bytes(payload)

    with pytest.raises(ingestion.IngestionStorageError, match="job_"):
        ingestion.incomplete_job_ids()










def test_ingestion_never_mutates_the_live_retrieval_corpus(monkeypatch):
    live_ids = [row["id"] for row in store.CHUNKS]
    created = ingestion.create_document(_sample_pdf(), "new.pdf")
    monkeypatch.setattr(ingestion, "page_text", lambda _page, _dpi: ("ข้อ 1 ข้อมูลใหม่", "text"))

    ingestion.process_job(created["job_id"])

    assert [row["id"] for row in store.CHUNKS] == live_ids
