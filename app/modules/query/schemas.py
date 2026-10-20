from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=400)
    history: list[str] = Field(default_factory=list, max_length=12)


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


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    request_id: UUID
    helpful: bool
    note: str = Field(default="", max_length=2000)
