from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class SessionContext(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    fund: str = Field(min_length=1, max_length=250)
    tier: str | None = Field(default=None, max_length=100)
    fiscal_year: int = Field(ge=2400, le=3000)
    year_assumed: bool = True
    intent: str = Field(max_length=40)


class Processing(BaseModel):
    analysis_method: str
    slot_attempted: bool = False
    slot_model: str | None = None
    slot_fallback_reason: str | None = None
    retrieval_method: str
    retrieval_fallback_reason: str | None = None
    answer_attempted: bool = False
    answer_model: str | None = None
    answer_fallback_reason: str | None = None
    corpus_revision: int


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=400)
    history: list[Annotated[str, Field(max_length=400)]] = Field(default_factory=list, max_length=8)
    context: SessionContext | None = None


class Citation(BaseModel):
    document_id: UUID
    version_id: UUID
    page: int
    chunk_id: UUID | None
    quote: str = ""
    label: str
    clause: str = ""
    url: str


class Verification(BaseModel):
    ok: bool
    problems: list[str]
    candidate_problems: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    request_id: UUID
    status: Literal["answered", "clarification_required", "insufficient_evidence", "blocked"]
    answer: str
    answer_source: Literal["llm", "template", "blocked"]
    citations: list[Citation]
    verification: Verification
    timings_ms: dict[str, float]
    context: SessionContext | None = None
    processing: Processing


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_id: UUID
    helpful: bool
    note: str = Field(default="", max_length=2000)
