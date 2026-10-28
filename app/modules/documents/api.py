"""HTTP validation and delegation only; database operations live in adapters."""
from urllib.parse import unquote
from uuid import UUID
from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from app.core.dependencies import Admin, Limit, Offset, services
from .schemas import (DocumentResponse, DocumentDetail, VersionResponse, PageResponse, ChunkResponse, UploadResponse, JobResponse)
from app.modules.knowledge.schemas import (Actor, PageEdit, ReviewDocument)

router = APIRouter(prefix="/api")


async def pdf_body(request):
    if request.headers.get("content-type", "").split(";")[0] != "application/pdf":
        raise HTTPException(415, "Use application/pdf")
    content = bytearray()
    async for part in request.stream():
        content.extend(part)
        if len(content) > request.app.state.settings.max_pdf_bytes:
            raise HTTPException(413, "PDF exceeds upload limit")
    return bytes(content)


@router.post("/documents", status_code=202, response_model=UploadResponse, tags=["Documents"],
             openapi_extra={"requestBody": {"required": True, "content": {"application/pdf": {
                 "schema": {"type": "string", "format": "binary"}}}}})
async def upload(request: Request, _: Admin, x_pdf_name: str = Header(default="document.pdf", max_length=500),
                 title: str | None = Query(default=None, max_length=250),
                 fiscal_year_be: int | None = Query(default=None, ge=2400, le=3000)):
    import asyncio
    content = await pdf_body(request)
    return await asyncio.to_thread(services(request).repository.create_document, content,
                                   unquote(x_pdf_name), title=title, fiscal_year_be=fiscal_year_be)


@router.post("/documents/inspect-pdf", tags=["Documents"])
async def inspect_pdf(request: Request, _: Admin, x_pdf_name: str = Header(default="document.pdf", max_length=500)):
    import asyncio
    from app.infrastructure.files.extraction import TesseractExtractor
    from app.modules.documents.pdf_inspection import amount_candidates
    content = await pdf_body(request)
    runtime = services(request)
    runtime.files.validate(content)
    pages = await asyncio.to_thread(TesseractExtractor().extract, content)
    return {"filename": unquote(x_pdf_name), "pages_total": len(pages), "saved": False, "verified": False,
            "note": "ผลการอ่านเป็นฉบับร่าง ไม่ได้บันทึกหรืออนุมัติข้อมูล",
            "pages": [{"page": p["number"], "text": p["text"][:8000], "method": p["method"],
                       "error": p["error"], "truncated": len(p["text"]) > 8000,
                       "amount_candidates": amount_candidates(p["text"])} for p in pages]}


@router.get("/documents", response_model=list[DocumentResponse], tags=["Documents"])
def documents(request: Request, _: Admin, limit: Limit = 50, offset: Offset = 0):
    return services(request).repository.list_documents(limit, offset)


@router.get("/documents/{document_id}", response_model=DocumentDetail, tags=["Documents"])
def document(document_id: UUID, request: Request, _: Admin):
    return services(request).repository.get_document(document_id)


@router.get("/documents/{document_id}/versions", response_model=list[VersionResponse], tags=["Documents"])
def versions(document_id: UUID, request: Request, _: Admin):
    return services(request).repository.get_document(document_id)["versions"]


@router.post("/documents/{document_id}/versions", status_code=202, response_model=UploadResponse, tags=["Documents"],
             openapi_extra={"requestBody": {"required": True, "content": {"application/pdf": {
                 "schema": {"type": "string", "format": "binary"}}}}})
async def upload_version(document_id: UUID, request: Request, _: Admin,
                         x_pdf_name: str = Header(default="document.pdf", max_length=500)):
    import asyncio
    content = await pdf_body(request)
    return await asyncio.to_thread(services(request).repository.create_document, content,
                                   unquote(x_pdf_name), document_id=document_id)


@router.get("/versions/{version_id}", response_model=VersionResponse, tags=["Documents"])
def version(version_id: UUID, request: Request, _: Admin):
    return services(request).repository.get_version(version_id)


@router.get("/versions/{version_id}/pages", response_model=list[PageResponse], tags=["Documents"])
def pages(version_id: UUID, request: Request, _: Admin):
    return services(request).repository.pages(version_id)


@router.get("/versions/{version_id}/chunks", response_model=list[ChunkResponse], tags=["Documents"])
def chunks(version_id: UUID, request: Request, _: Admin):
    return services(request).repository.chunks(version_id)


@router.patch("/versions/{version_id}/pages/{page}", response_model=PageResponse, tags=["Documents"])
def edit(version_id: UUID, page: int, body: PageEdit, request: Request, _: Admin):
    return services(request).repository.edit_page(version_id, page, body)


@router.post("/versions/{version_id}/review", response_model=VersionResponse, tags=["Documents"])
def review(version_id: UUID, body: ReviewDocument, request: Request, _: Admin):
    return services(request).repository.review_document(version_id, body)


@router.post("/versions/{version_id}/publish", response_model=VersionResponse, tags=["Documents"])
def publish(version_id: UUID, body: Actor, request: Request, _: Admin):
    return services(request).repository.publish(version_id, body.reviewer)


@router.post("/versions/{version_id}/withdraw", response_model=VersionResponse, tags=["Documents"])
def withdraw(version_id: UUID, body: Actor, request: Request, _: Admin):
    return services(request).repository.withdraw(version_id, body.reviewer)


@router.get("/versions/{version_id}/source", response_class=FileResponse, tags=["Documents"])
def private_source(version_id: UUID, request: Request, _: Admin):
    path, filename = services(request).repository.source(version_id, admin=True)
    return FileResponse(path, media_type="application/pdf", filename=filename,
                        content_disposition_type="inline", headers={"Cache-Control": "no-store"})


@router.get("/sources/{version_id}/pages/{page}", response_class=FileResponse, tags=["Documents"])
def source(version_id: UUID, page: int, request: Request):
    path, filename = services(request).repository.source(version_id, page=page)
    return FileResponse(path, media_type="application/pdf", filename=filename,
                        content_disposition_type="inline", headers={"Cache-Control": "no-store"})


@router.get("/jobs", response_model=list[JobResponse], tags=["Jobs"])
def jobs(request: Request, _: Admin, limit: Limit = 50, offset: Offset = 0):
    return services(request).repository.jobs(limit, offset)


@router.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
def job(job_id: UUID, request: Request, _: Admin):
    return services(request).repository.get_job(job_id)


@router.post("/jobs/{job_id}/retry", response_model=JobResponse, status_code=202, tags=["Jobs"])
def retry(job_id: UUID, request: Request, _: Admin):
    return services(request).repository.retry(job_id)


# The two knowledge resources share a contract, but have separate Swagger operations.


