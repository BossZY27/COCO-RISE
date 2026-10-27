from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceInput(Input):
    version_id: UUID
    page: int = Field(ge=1)
    chunk_id: UUID | None = None
    quote: str = Field(min_length=1, max_length=12000)
    clause: str = Field(default="", max_length=160)


class Condition(Input):
    field: str = Field(pattern=r"^(staff_type|department|grants_this_year(?:\..+)?)$")
    op: Literal["in", "eq", "lt", "not_received"]
    value: Any
    fail_reason: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def valid_operand(self):
        if self.op in {"in", "not_received"} and (not isinstance(self.value, list) or
                not self.value or any(not isinstance(v, str) for v in self.value)):
            raise ValueError("This operator needs a nonempty list of strings")
        if self.op == "lt" and (type(self.value) is not int or self.value < 0):
            raise ValueError("lt needs a nonnegative integer")
        if self.op == "eq" and not isinstance(self.value, str):
            raise ValueError("eq needs a string")
        if self.op == "not_received" and self.field != "grants_this_year":
            raise ValueError("not_received only applies to grants_this_year")
        return self


class KnowledgeInput(Input):
    fund: str = Field(min_length=1, max_length=250)
    fiscal_year_be: int | None = Field(default=None, ge=2400, le=3000)
    tier: str | None = Field(default=None, max_length=100)
    amount_type: Literal["ceiling", "allocated"] | None = None
    amount: Decimal | None = Field(default=None, ge=0, max_digits=14, decimal_places=2)
    total_cap: Decimal | None = Field(default=None, ge=0, max_digits=14, decimal_places=2)
    max_grants_per_person: int | None = Field(default=None, ge=1)
    max_grants_per_year: int | None = Field(default=None, ge=1)
    note: str | None = Field(default=None, max_length=4000)
    conditions: list[Condition] = Field(default_factory=list, max_length=40)
    evidence: list[EvidenceInput] = Field(min_length=1, max_length=30)


class Decision(Input):
    reviewer: str = Field(min_length=1, max_length=120)
    decision: Literal["approved", "rejected"]


class ReviewDocument(Input):
    reviewer: str = Field(min_length=1, max_length=120)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    confirmed_pages: list[int] = Field(min_length=1, max_length=100)


class PageEdit(Input):
    text: str = Field(min_length=1, max_length=100000)
    reviewer: str = Field(min_length=1, max_length=120)


class Actor(Input):
    reviewer: str = Field(min_length=1, max_length=120)


class Record(BaseModel):
    model_config = ConfigDict(extra="allow", from_attributes=True)
    id: UUID


class EvidenceResponse(BaseModel):
    id: UUID
    version_id: UUID
    page: int
    chunk_id: UUID | None
    quote: str
    clause: str


class KnowledgeResponse(BaseModel):
    id: UUID
    legacy_id: str | None
    kind: Literal["fact", "rule"]
    fund: str
    program_id: UUID
    call_id: UUID | None
    tier_id: UUID | None
    fiscal_year_be: int | None
    amount_type: Literal["ceiling", "allocated"] | None
    amount: Decimal | None
    status: Literal["draft", "approved", "rejected"]
    legacy_document_gate: bool
    payload: dict[str, Any]
    reviewed_by: str | None
    reviewed_at: datetime | None
    evidence: list[EvidenceResponse]
