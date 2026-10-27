"""Persistent state. Legacy JSON payloads retain provenance, never act as runtime storage."""
import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (BigInteger, Boolean, CheckConstraint, DateTime, ForeignKey, Index,
                        Integer, Numeric, String, Text, UniqueConstraint, func, text)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Identified:
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Program(Identified, Base):
    __tablename__ = "programs"
    __table_args__ = {"schema": "core"}
    name: Mapped[str] = mapped_column(Text, unique=True)


class Call(Identified, Base):
    __tablename__ = "annual_calls"
    __table_args__ = (UniqueConstraint("program_id", "fiscal_year_be"), {"schema": "core"})
    program_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("core.programs.id"))
    fiscal_year_be: Mapped[int] = mapped_column(Integer)


class Tier(Identified, Base):
    __tablename__ = "tiers"
    __table_args__ = {"schema": "core"}
    name: Mapped[str] = mapped_column(Text, unique=True)


class Alias(Identified, Base):
    __tablename__ = "aliases"
    __table_args__ = {"schema": "core"}
    name: Mapped[str] = mapped_column(Text, unique=True)
    programs: Mapped[list] = mapped_column(JSONB)


class Persona(Identified, Base):
    __tablename__ = "personas"
    __table_args__ = {"schema": "core"}
    legacy_id: Mapped[str] = mapped_column(String(128), unique=True)
    payload: Mapped[dict] = mapped_column(JSONB)


class Document(Identified, Base):
    __tablename__ = "documents"
    __table_args__ = {"schema": "knowledge"}
    legacy_id: Mapped[str | None] = mapped_column(Text, unique=True)
    title: Mapped[str] = mapped_column(Text)
    fiscal_year_be: Mapped[int | None] = mapped_column(Integer)


class Version(Identified, Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "number"),
                      UniqueConstraint("document_id", "sha256"),
                      Index("one_current_document_version", "document_id", unique=True,
                            postgresql_where=text("current")),
                      CheckConstraint("status in ('draft','reviewed','published','withdrawn','superseded')"),
                      {"schema": "knowledge"})
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("knowledge.documents.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str | None] = mapped_column(String(64))
    filename: Mapped[str] = mapped_column(Text)
    file_key: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="draft")
    current: Mapped[bool] = mapped_column(Boolean, default=False)
    pages_total: Mapped[int] = mapped_column(Integer, default=0)
    reviewed_by: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    provenance: Mapped[dict] = mapped_column(JSONB, default=dict)


class Page(Identified, Base):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("version_id", "number"), CheckConstraint("number > 0"), {"schema": "knowledge"})
    version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("knowledge.document_versions.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    method: Mapped[str] = mapped_column(String(24))
    error: Mapped[str | None] = mapped_column(Text)
    reviewed_by: Mapped[str | None] = mapped_column(Text)


class ChunkSet(Identified, Base):
    __tablename__ = "chunk_sets"
    __table_args__ = {"schema": "knowledge"}
    version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("knowledge.document_versions.id"), unique=True)
    strategy: Mapped[str] = mapped_column(Text, default="structured-900-v1")


class Chunk(Identified, Base):
    __tablename__ = "chunks"
    __table_args__ = (Index("chunks_version_page", "version_id", "page"),
                      CheckConstraint("page > 0"), {"schema": "knowledge"})
    legacy_id: Mapped[str | None] = mapped_column(Text, unique=True)
    version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("knowledge.document_versions.id"))
    chunk_set_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("knowledge.chunk_sets.id"))
    page: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)


class Evidence(Identified, Base):
    __tablename__ = "evidence"
    __table_args__ = (CheckConstraint("page > 0"), {"schema": "knowledge"})
    version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("knowledge.document_versions.id"), index=True)
    page: Mapped[int] = mapped_column(Integer)
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("knowledge.chunks.id"))
    quote: Mapped[str] = mapped_column(Text)
    clause: Mapped[str] = mapped_column(Text, default="")


class Knowledge(Identified, Base):
    __tablename__ = "knowledge_items"
    __table_args__ = (CheckConstraint("kind in ('fact','rule')"),
                      CheckConstraint("status in ('draft','approved','rejected')"),
                      CheckConstraint("amount_type is null or amount_type in ('ceiling','allocated')"),
                      CheckConstraint("amount is null or amount >= 0"),
                      CheckConstraint("amount_type != 'allocated' or fiscal_year_be is not null"),
                      Index("knowledge_lookup", "program_id", "fiscal_year_be", "amount_type", "status"),
                      {"schema": "core"})
    legacy_id: Mapped[str | None] = mapped_column(Text, unique=True)
    kind: Mapped[str] = mapped_column(String(8))
    program_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("core.programs.id"))
    call_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("core.annual_calls.id"))
    tier_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("core.tiers.id"))
    fiscal_year_be: Mapped[int | None] = mapped_column(Integer)
    amount_type: Mapped[str | None] = mapped_column(String(16))
    amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    status: Mapped[str] = mapped_column(String(16), default="draft")
    # Legacy rules are gated by the original document review policy, not a fabricated approval.
    legacy_document_gate: Mapped[bool] = mapped_column(Boolean, default=False)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    reviewed_by: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnowledgeEvidence(Base):
    __tablename__ = "knowledge_evidence"
    __table_args__ = {"schema": "knowledge"}
    knowledge_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("core.knowledge_items.id"), primary_key=True)
    evidence_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("knowledge.evidence.id"), primary_key=True)


class CorpusRevision(Base):
    __tablename__ = "corpus_revision"
    __table_args__ = {"schema": "retrieval"}
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    revision: Mapped[int] = mapped_column(BigInteger, default=0)


class EmbeddingModel(Identified, Base):
    __tablename__ = "embedding_models"
    __table_args__ = {"schema": "retrieval"}
    name: Mapped[str] = mapped_column(Text)
    revision: Mapped[str] = mapped_column(Text, unique=True)
    dimensions: Mapped[int] = mapped_column(Integer, default=1024)


class Generation(Identified, Base):
    __tablename__ = "index_generations"
    __table_args__ = {"schema": "retrieval"}
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("retrieval.embedding_models.id"))
    corpus_revision: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(16), default="building")


class Job(Identified, Base):
    __tablename__ = "ingestion_jobs"
    __table_args__ = (CheckConstraint("attempts >= 0"), Index("claim_jobs", "status", "lease_until"), {"schema": "audit"})
    legacy_id: Mapped[str | None] = mapped_column(Text, unique=True)
    version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("knowledge.document_versions.id"))
    kind: Mapped[str] = mapped_column(String(16), default="ingest")
    status: Mapped[str] = mapped_column(String(24), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    owner: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)


class Event(Identified, Base):
    __tablename__ = "events"
    __table_args__ = {"schema": "audit"}
    actor: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)


class ImportRun(Identified, Base):
    __tablename__ = "import_runs"
    __table_args__ = {"schema": "audit"}
    checksum: Mapped[str] = mapped_column(String(64))
    manifest: Mapped[dict] = mapped_column(JSONB)


class QueryRun(Identified, Base):
    __tablename__ = "query_runs"
    __table_args__ = {"schema": "audit"}
    question: Mapped[str] = mapped_column(Text)
    response: Mapped[dict] = mapped_column(JSONB)
    model_revision: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(Text, default="rise-grounded-v1")


class Feedback(Identified, Base):
    __tablename__ = "feedback"
    __table_args__ = {"schema": "audit"}
    request_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("audit.query_runs.id"), index=True)
    helpful: Mapped[bool] = mapped_column(Boolean)
    note: Mapped[str] = mapped_column(Text, default="")
