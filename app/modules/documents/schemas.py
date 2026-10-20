from datetime import datetime
from typing import Any, Literal
from uuid import UUID
from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: UUID
    created_at: datetime
    legacy_id: str | None
    title: str
    fiscal_year_be: int | None


class VersionResponse(BaseModel):
    id: UUID
    created_at: datetime
    document_id: UUID
    number: int
    sha256: str | None
    filename: str
    status: Literal["draft", "reviewed", "published", "withdrawn", "superseded"]
    current: bool
    pages_total: int
    reviewed_by: str | None
    reviewed_at: datetime | None
    provenance: dict[str, Any]


class DocumentDetail(DocumentResponse):
    versions: list[VersionResponse]


class PageResponse(BaseModel):
    id: UUID
    version_id: UUID
    number: int
    text: str
    method: str
    error: str | None
    reviewed_by: str | None


class ChunkResponse(BaseModel):
    id: UUID
    legacy_id: str | None
    version_id: UUID
    chunk_set_id: UUID
    page: int
    text: str
    verified: bool
    payload: dict[str, Any]


class UploadResponse(BaseModel):
    document_id: UUID
    version_id: UUID
    job_id: UUID | None
    duplicate: bool
    status: str


class JobResponse(BaseModel):
    id: UUID
    created_at: datetime
    legacy_id: str | None
    version_id: UUID | None
    kind: Literal["ingest", "index"]
    status: Literal["queued", "running", "awaiting_review", "needs_ocr", "done", "failed"]
    attempts: int
    owner: UUID | None
    lease_until: datetime | None
    heartbeat_at: datetime | None
    error: str | None
    payload: dict[str, Any]
