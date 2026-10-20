"""Administrative API: PDF quarantine, human review, and published evidence.

Uploads remain drafts until a named administrator confirms every page.
"""

import asyncio
import os
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, Body, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from app.infrastructure.json import audit
from app.modules.documents import pdf_inspection, publication, review
from app.workers import ingestion
from .schemas import (
    IngestionAcceptedResponse, IngestionChunkListResponse, IngestionChunkPatch,
    IngestionChunkResponse, IngestionDocumentListResponse, IngestionDocumentResponse,
    IngestionJobResponse, PublicationRequest,
)

ADMIN_ROUTES_ENABLED = os.getenv("RISE_ADMIN_ENABLED", "on").strip().lower() in {"1", "on", "true", "yes"}
router = APIRouter()


def _admin(token: str | None) -> None:
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        import hmac
        expected = runtime.settings.admin_token
        if not expected or not token or not hmac.compare_digest(token, expected):
            raise HTTPException(status_code=401, detail="Valid X-Admin-Token required")
        return
    if not ADMIN_ROUTES_ENABLED or not review.enabled():
        raise HTTPException(status_code=404, detail="หน้าผู้ดูแลปิดอยู่ ตั้ง RISE_ADMIN_TOKEN ก่อนจึงจะเปิด")
    if not review.authorised(token):
        raise HTTPException(status_code=401, detail="โทเคนผู้ดูแลไม่ถูกต้อง")


async def _read_pdf_upload(request: Request) -> bytes:
    """Use the same content-type and size checks for preview and ingestion."""
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/pdf":
        raise HTTPException(status_code=415, detail="กรุณาส่งไฟล์ชนิด application/pdf")
    data = bytearray()
    async for part in request.stream():
        data.extend(part)
        if len(data) > pdf_inspection.MAX_BYTES:
            raise HTTPException(status_code=413, detail="ไฟล์ PDF ใหญ่เกิน 15 MB")
    return bytes(data)


@router.get("/v1/recent")
def recent(limit: int = 20, x_admin_token: str | None = Header(default=None)):
    """What people actually asked. The fund officer reads this, and the good ones become test cases."""
    _admin(x_admin_token)
    return audit.tail(limit)


@router.get("/v1/admin/pending")
def admin_pending(x_admin_token: str | None = Header(default=None)):
    _admin(x_admin_token)
    return review.pending()


@router.get("/v1/admin/source")
def admin_source(doc: str, x_admin_token: str | None = Header(default=None)):
    _admin(x_admin_token)
    path = review.source_document(doc)
    if path is None:
        raise HTTPException(status_code=404, detail="ไม่พบเอกสารต้นฉบับของประกาศนี้")
    return FileResponse(path, media_type="application/pdf", filename=path.name,
                        content_disposition_type="inline",
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


@router.post("/v1/admin/inspect-pdf")
async def admin_inspect_pdf(request: Request, x_admin_token: str | None = Header(default=None),
                            x_pdf_name: str = Header(default="document.pdf")):
    """Preview an uploaded PDF; never store it or mark its contents as verified."""
    _admin(x_admin_token)
    data = await _read_pdf_upload(request)
    try:
        return await asyncio.to_thread(
            ingestion.run_ocr_serialized,
            pdf_inspection.inspect,
            data,
            unquote(x_pdf_name),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/v1/admin/documents", status_code=202, response_model=IngestionAcceptedResponse)
async def admin_create_document(request: Request,
                                x_admin_token: str | None = Header(default=None),
                                x_pdf_name: str = Header(default="document.pdf")):
    """Persist a PDF in quarantine and process it into draft chunks in the background."""
    _admin(x_admin_token)
    data = await _read_pdf_upload(request)
    try:
        accepted = await asyncio.to_thread(ingestion.create_document, data, unquote(x_pdf_name))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not accepted["duplicate"] or accepted["status"] in {"queued", "extracting", "chunking"}:
        ingestion.enqueue_job(accepted["job_id"])
    return accepted


@router.get("/v1/admin/jobs/{job_id}", response_model=IngestionJobResponse)
def admin_ingestion_job(job_id: str, x_admin_token: str | None = Header(default=None)):
    _admin(x_admin_token)
    job = ingestion.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ไม่พบงานนำเข้าเอกสาร")
    return job


@router.post(
    "/v1/admin/jobs/{job_id}/retry",
    status_code=202,
    response_model=IngestionJobResponse,
)
def admin_retry_ingestion_job(job_id: str,
                              x_admin_token: str | None = Header(default=None)):
    _admin(x_admin_token)
    try:
        job = ingestion.retry_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if job is None:
        raise HTTPException(status_code=404, detail="ไม่พบงานนำเข้าเอกสาร")
    ingestion.enqueue_job(job_id, after_current=True)
    return job


@router.get("/v1/admin/documents", response_model=IngestionDocumentListResponse)
def admin_documents(x_admin_token: str | None = Header(default=None), limit: int = 50,
                    offset: int = 0):
    _admin(x_admin_token)
    if limit < 1 or limit > 100 or offset < 0:
        raise HTTPException(status_code=400, detail="ช่วงรายการเอกสารไม่ถูกต้อง")
    all_documents = ingestion.list_documents()
    items = all_documents[offset:offset + limit]
    return {"items": items, "count": len(all_documents), "limit": limit, "offset": offset}


@router.get("/v1/admin/documents/{document_id}", response_model=IngestionDocumentResponse)
def admin_document(document_id: str, x_admin_token: str | None = Header(default=None)):
    _admin(x_admin_token)
    document = ingestion.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="ไม่พบเอกสารนำเข้า")
    return document


@router.get("/v1/admin/documents/{document_id}/source")
def admin_ingestion_source(document_id: str, x_admin_token: str | None = Header(default=None)):
    _admin(x_admin_token)
    path = ingestion.source_path(document_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="ไม่พบ PDF ต้นฉบับ")
    return FileResponse(path, media_type="application/pdf", filename=path.name,
                        content_disposition_type="inline",
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


@router.get(
    "/v1/admin/documents/{document_id}/chunks",
    response_model=IngestionChunkListResponse,
)
def admin_ingestion_chunks(document_id: str, x_admin_token: str | None = Header(default=None),
                           limit: int = 200, offset: int = 0):
    _admin(x_admin_token)
    if ingestion.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="ไม่พบเอกสารนำเข้า")
    if limit < 1 or limit > 500 or offset < 0:
        raise HTTPException(status_code=400, detail="ช่วงรายการข้อความไม่ถูกต้อง")
    all_chunks = ingestion.list_chunks(document_id)
    return {"items": all_chunks[offset:offset + limit], "count": len(all_chunks),
            "limit": limit, "offset": offset}


@router.patch(
    "/v1/admin/documents/{document_id}/chunks/{chunk_id}",
    response_model=IngestionChunkResponse,
)
def admin_correct_ingestion_chunk(document_id: str, chunk_id: str, body: IngestionChunkPatch,
                                  x_admin_token: str | None = Header(default=None)):
    _admin(x_admin_token)
    changes = body.model_dump(exclude_none=True)
    reviewer = changes.pop("reviewer")
    try:
        chunk = ingestion.update_chunk(document_id, chunk_id, reviewer, changes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if chunk is None:
        raise HTTPException(status_code=404, detail="ไม่พบข้อความที่ต้องการแก้")
    return chunk


@router.post("/v1/admin/documents/{document_id}/publish")
def admin_publish_document(document_id: str, body: PublicationRequest,
                           x_admin_token: str | None = Header(default=None)):
    """Publish reviewed text only; monetary facts and rules remain separate."""
    _admin(x_admin_token)
    if ingestion.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="ไม่พบเอกสารนำเข้า")
    try:
        return publication.publisher.publish(document_id, body.reviewer, body.source_sha256,
                                             body.confirmed_pages)
    except publication.PublicationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/v1/admin/verify-fact")
def admin_verify_fact(x_admin_token: str | None = Header(default=None), payload: dict = Body(...)):
    _admin(x_admin_token)
    result = review.verify_fact(payload.get("id", ""), payload.get("reviewer", ""),
                                payload.get("amount"))
    if result.get("code") == "already_verified":
        raise HTTPException(status_code=409, detail=result["error"])
    return result


@router.post("/v1/admin/verify-doc")
def admin_verify_doc(x_admin_token: str | None = Header(default=None), payload: dict = Body(...)):
    _admin(x_admin_token)
    return review.verify_chunks(payload.get("doc", ""), payload.get("reviewer", ""))


ADMIN_PAGE = (Path(__file__).resolve().parents[2] / "static" / "admin.html").read_text(encoding="utf-8")


@router.get("/admin", response_class=HTMLResponse)
def admin_page():
    if not ADMIN_ROUTES_ENABLED or not review.enabled():
        return HTMLResponse("<p>หน้าผู้ดูแลปิดอยู่ ตั้งตัวแปร RISE_ADMIN_TOKEN แล้วเปิดเซิร์ฟเวอร์ใหม่</p>",
                            status_code=404)
    return ADMIN_PAGE
