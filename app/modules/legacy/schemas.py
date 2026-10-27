"""Typed request and response contracts for the public and administrator APIs."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core import security as limits


HistoryTurn = Annotated[
    str,
    Field(max_length=limits.MAX_HISTORY_ITEM_CHARS),
]


class SessionContext(BaseModel):
    """The last resolved topic in one browser chat, never an authority for facts."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    fund: str = Field(min_length=1, max_length=160)
    tier: str | None = Field(default=None, max_length=40)
    fiscal_year: int = Field(ge=2500, le=2700)
    year_assumed: bool = True
    intent: str = Field(max_length=40)


class Query(BaseModel):
    """ถามข้อมูลทุนในเดโม ใช้ persona_id สำหรับข้อมูลผู้ใช้สมมติเท่านั้น"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=limits.MAX_QUESTION_CHARS,
                          description="คำถามภาษาไทยเรื่องทุนวิจัย")
    persona_id: str = Field(default="P1", min_length=1, max_length=128,
                            description="รหัส persona สมมติในเดโม")
    history: list[HistoryTurn] = Field(
        default_factory=list,
        max_length=limits.MAX_HISTORY_TURNS,
    )
    model: str | None = Field(default=None, min_length=1, max_length=200,
                              description="คงไว้เพื่อ compatibility; safe mode จะปฏิเสธค่าที่ไม่ว่าง")
    context: SessionContext | None = Field(
        default=None,
        description="บริบทที่สรุปแล้วของการสนทนาปัจจุบันในเบราว์เซอร์",
    )


class Feedback(BaseModel):
    """A small user report suitable for the append-only audit log."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(
        min_length=1,
        max_length=limits.MAX_FEEDBACK_QUESTION_CHARS,
        description="คำถามที่ต้องการให้ความเห็น",
    )
    answer: str = Field(min_length=1, max_length=limits.MAX_FEEDBACK_ANSWER_CHARS,
                        description="คำตอบที่แสดงในเดโม")
    helpful: bool = Field(description="true เมื่อคำตอบมีประโยชน์")
    note: str = Field(default="", max_length=limits.MAX_FEEDBACK_NOTE_CHARS,
                      description="หมายเหตุเพิ่มเติม; เก็บเมื่อเปิด audit")


SECTION_TYPES = Literal[
    "title",
    "preamble",
    "definition",
    "fund_list",
    "amount_table",
    "eligibility",
    "application",
    "timeline",
    "contact",
    "other",
]

INGESTION_STATUSES = Literal[
    "queued",
    "extracting",
    "chunking",
    "awaiting_review",
    "needs_ocr",
    "failed",
]


class IngestionAcceptedResponse(BaseModel):
    document_id: str
    job_id: str
    status: INGESTION_STATUSES
    duplicate: bool


class IngestionJobResponse(BaseModel):
    job_id: str
    document_id: str
    status: INGESTION_STATUSES
    attempts: int = 0
    pages_total: int = 0
    pages_processed: int = 0
    chunk_count: int = 0
    created_at: str
    updated_at: str
    started_at: str | None = None
    processed_at: str | None = None
    retry_requested_at: str | None = None
    error: str | None = None
    needs_ocr_pages: list[int] = Field(default_factory=list)


class IngestionDocumentResponse(BaseModel):
    document_id: str
    job_id: str
    filename: str
    content_sha256: str
    content_bytes: int
    pages_total: int
    pages_processed: int = 0
    chunk_count: int = 0
    status: INGESTION_STATUSES
    verified: bool
    published: bool
    created_at: str
    updated_at: str
    processed_at: str | None = None
    reconciled_at: str | None = None
    last_edited_by: str | None = None
    last_edited_at: str | None = None
    error: str | None = None
    needs_ocr_pages: list[int] = Field(default_factory=list)


class IngestionChunkResponse(BaseModel):
    id: str
    document_id: str
    doc: str
    fiscal_year: int | None = None
    source_filename: str
    text: str
    page: int
    extraction_method: str
    order_in_page: int
    global_order: int
    clause: str
    parent_clause: str | None = None
    part_index: int
    section_type: SECTION_TYPES
    section_label: str
    amount_candidates: list[str] = Field(default_factory=list)
    status: Literal["draft", "active"]
    verified: bool
    published: bool
    created_at: str
    updated_at: str
    last_edited_by: str | None = None
    last_edited_at: str | None = None
    edit_history: list[dict[str, Any]] = Field(default_factory=list)


class IngestionDocumentListResponse(BaseModel):
    items: list[IngestionDocumentResponse]
    count: int
    limit: int
    offset: int


class IngestionChunkListResponse(BaseModel):
    items: list[IngestionChunkResponse]
    count: int
    limit: int
    offset: int


class IngestionChunkPatch(BaseModel):
    """A human correction to a quarantined chunk, never an approval action."""

    model_config = ConfigDict(extra="forbid")

    reviewer: str = Field(min_length=1, max_length=120)
    text: str | None = Field(default=None, min_length=1, max_length=12_000)
    clause: str | None = Field(default=None, max_length=120)
    section_type: SECTION_TYPES | None = None
    section_label: str | None = Field(default=None, max_length=160)

    @field_validator("reviewer")
    @classmethod
    def reviewer_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("กรุณาระบุชื่อผู้ตรวจ")
        return value


class PublicationRequest(BaseModel):
    """An explicit, page-by-page attestation by the PDF reviewer."""

    model_config = ConfigDict(extra="forbid")

    reviewer: str = Field(min_length=1, max_length=120)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    confirmed_pages: list[int]
