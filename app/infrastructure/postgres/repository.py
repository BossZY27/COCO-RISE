"""Short transactions for document review, structured knowledge, and leased jobs."""
import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from sqlalchemy import and_, func, or_, select, update, delete, text
from app.core.errors import ServiceError
from app.modules.documents.chunking import split_structured_page
from .models import (Alias, Call, Chunk, ChunkSet, CorpusRevision, Document, Event, Evidence,
                     Feedback, Job, Knowledge, KnowledgeEvidence, Page, Persona, Program,
                     QueryRun, Tier, Version)


def now():
    return datetime.now(timezone.utc)


def record(row):
    return {c.name: getattr(row, c.name) for c in row.__table__.columns}


def require(session, model, identifier, lock=False):
    statement = select(model).where(model.id == identifier)
    if lock:
        statement = statement.with_for_update()
    row = session.scalar(statement)
    if row is None:
        raise ServiceError(404, f"{model.__name__} not found")
    return row


def revision(session, bump=False):
    if bump:
        return session.execute(update(CorpusRevision).where(CorpusRevision.id == 1)
                               .values(revision=CorpusRevision.revision + 1)
                               .returning(CorpusRevision.revision)).scalar_one()
    return session.scalar(select(CorpusRevision.revision).where(CorpusRevision.id == 1))


def event(session, actor, action, target, **payload):
    session.add(Event(actor=actor, action=action, target_id=target, payload=payload))


def named(session, model, name):
    row = session.scalar(select(model).where(model.name == name))
    if row is None:
        row = model(name=name)
        session.add(row)
        session.flush()
    return row


def chunk_page(session, version, page, chunk_set):
    for part in split_structured_page(page.text, page.number, page.method):
        session.add(Chunk(version_id=version.id, chunk_set_id=chunk_set.id,
                          page=page.number, text=part["text"], verified=False, payload=part))


class Repository:
    def __init__(self, database, files, settings):
        self.db, self.files, self.settings = database, files, settings

    def list_documents(self, limit=50, offset=0):
        with self.db.transaction() as s:
            return [record(r) for r in s.scalars(select(Document).order_by(Document.created_at, Document.id)
                                                 .limit(limit).offset(offset))]

    def get_document(self, document_id):
        with self.db.transaction() as s:
            document = require(s, Document, document_id)
            return {**record(document), "versions": [record(v) for v in s.scalars(
                select(Version).where(Version.document_id == document_id).order_by(Version.number))]}

    def get_version(self, version_id):
        with self.db.transaction() as s:
            return record(require(s, Version, version_id))

    def create_document(self, content, filename, document_id=None, title=None, fiscal_year_be=None):
        page_count = self.files.validate(content)
        digest = hashlib.sha256(content).hexdigest()
        filename = filename.replace("\\", "/").rsplit("/", 1)[-1][:250] or "document.pdf"
        with self.db.transaction() as s:
            # Serialize uploads before duplicate detection and version-number allocation.
            s.execute(text("SELECT pg_advisory_xact_lock(7249301)"))
            if document_id is not None:
                doc = require(s, Document, document_id, lock=True)
            else:
                existing = s.scalar(select(Version).where(Version.sha256 == digest).order_by(Version.created_at))
                if existing:
                    job = s.scalar(select(Job).where(Job.version_id == existing.id).order_by(Job.created_at))
                    return {"document_id": existing.document_id, "version_id": existing.id,
                            "job_id": job.id if job else None, "duplicate": True, "status": existing.status}
                count = s.scalar(select(func.count()).select_from(Document))
                if count >= self.settings.ingestion_max_documents:
                    raise ServiceError(409, "Document quota reached")
                doc = Document(title=title or filename, fiscal_year_be=fiscal_year_be)
                s.add(doc)
                s.flush()
            existing = s.scalar(select(Version).where(Version.document_id == doc.id, Version.sha256 == digest))
            if existing:
                job = s.scalar(select(Job).where(Job.version_id == existing.id))
                return {"document_id": doc.id, "version_id": existing.id, "job_id": job.id if job else None,
                        "duplicate": True, "status": existing.status}
            used_bytes = sum(p.stat().st_size for p in self.files.root.glob("*.pdf") if p.is_file())
            destination = self.files.root / (digest + ".pdf")
            additional = 0 if destination.exists() else len(content)
            if used_bytes + additional > self.settings.ingestion_max_total_bytes:
                raise ServiceError(409, "PDF storage quota reached")
            digest, key = self.files.save(content)
            number = (s.scalar(select(func.max(Version.number)).where(Version.document_id == doc.id)) or 0) + 1
            version = Version(document_id=doc.id, number=number, sha256=digest, file_key=key,
                              filename=filename, pages_total=page_count)
            s.add(version)
            s.flush()
            s.add(ChunkSet(version_id=version.id))
            job = Job(version_id=version.id)
            s.add(job)
            s.flush()
            event(s, "admin", "upload", version.id, sha256=digest)
            return {"document_id": doc.id, "version_id": version.id, "job_id": job.id,
                    "duplicate": False, "status": "queued"}

    def source(self, version_id, admin=False, page=None):
        with self.db.transaction() as s:
            version = require(s, Version, version_id)
            # Published history stays addressable; drafts never become public by guessing IDs.
            if not admin and version.status not in {"published", "superseded", "withdrawn"}:
                raise ServiceError(404, "Published source not found")
            if page is not None and not 1 <= page <= version.pages_total:
                raise ServiceError(404, "Page not found")
            self.files.checked(version.file_key, version.sha256)
            return self.files.path(version.file_key), version.filename

    def pages(self, version_id):
        with self.db.transaction() as s:
            require(s, Version, version_id)
            return [record(p) for p in s.scalars(select(Page).where(Page.version_id == version_id).order_by(Page.number))]

    def chunks(self, version_id):
        with self.db.transaction() as s:
            require(s, Version, version_id)
            return [record(c) for c in s.scalars(select(Chunk).where(Chunk.version_id == version_id).order_by(Chunk.page, Chunk.id))]

    def edit_page(self, version_id, number, body):
        with self.db.transaction() as s:
            version = require(s, Version, version_id, lock=True)
            if version.status not in {"draft", "reviewed"}:
                raise ServiceError(409, "Create a new version to edit published text")
            if s.scalar(select(Job).where(Job.version_id == version_id, Job.status.in_(["queued", "running"]))):
                raise ServiceError(409, "Ingestion must finish before editing")
            page = s.scalar(select(Page).where(Page.version_id == version_id, Page.number == number))
            if page is None:
                raise ServiceError(404, "Page not found")
            # Existing evidence pins chunks. Deleting them would silently change reviewed claims.
            ids = select(Chunk.id).where(Chunk.version_id == version_id, Chunk.page == number)
            if s.scalar(select(Evidence.id).where(Evidence.chunk_id.in_(ids)).limit(1)):
                raise ServiceError(409, "Page is referenced by knowledge; create a new document version")
            page.text, page.error, page.reviewed_by = body.text, None, None
            version.status, version.reviewed_by, version.reviewed_at = "draft", None, None
            s.execute(delete(Chunk).where(Chunk.version_id == version_id, Chunk.page == number))
            chunk_set = s.scalar(select(ChunkSet).where(ChunkSet.version_id == version_id))
            chunk_page(s, version, page, chunk_set)
            for job in s.scalars(select(Job).where(Job.version_id == version_id)):
                job.payload = {**job.payload, "human_edited": True}
            event(s, body.reviewer, "edit_page", page.id)
            return record(page)

    def review_document(self, version_id, body):
        with self.db.transaction() as s:
            version = require(s, Version, version_id, lock=True)
            if version.status != "draft":
                raise ServiceError(409, "Only a draft can be reviewed")
            if body.source_sha256 != version.sha256:
                raise ServiceError(409, "Source checksum mismatch")
            self.files.checked(version.file_key, version.sha256)
            expected = set(range(1, version.pages_total + 1))
            if set(body.confirmed_pages) != expected or len(body.confirmed_pages) != len(expected):
                raise ServiceError(409, "Confirm every PDF page exactly once")
            pages = list(s.scalars(select(Page).where(Page.version_id == version_id)))
            if ({p.number for p in pages} != expected or any(p.error or not p.text.strip() for p in pages)
                    or s.scalar(select(Job.id).where(Job.version_id == version_id, Job.status.in_(["running", "queued"])))):
                raise ServiceError(409, "Every page must have complete, corrected text")
            version.status, version.reviewed_by, version.reviewed_at = "reviewed", body.reviewer, now()
            for p in pages:
                p.reviewed_by = body.reviewer
            s.execute(update(Chunk).where(Chunk.version_id == version_id).values(verified=True))
            event(s, body.reviewer, "review_document", version.id)
            return record(version)

    def publish(self, version_id, actor):
        with self.db.transaction() as s:
            version = require(s, Version, version_id)
            require(s, Document, version.document_id, lock=True)
            version = require(s, Version, version_id, lock=True)
            if version.status != "reviewed":
                raise ServiceError(409, "Review every page before publication")
            self.files.checked(version.file_key, version.sha256)
            s.execute(update(Version).where(Version.document_id == version.document_id, Version.current.is_(True))
                      .values(current=False, status="superseded"))
            s.flush()
            version.status, version.current = "published", True
            revision(s, True)
            s.add(Job(kind="index", payload={}))
            event(s, actor, "publish", version.id)
            return record(version)

    def withdraw(self, version_id, actor):
        with self.db.transaction() as s:
            version = require(s, Version, version_id, lock=True)
            if not version.current:
                raise ServiceError(409, "Version is not current")
            version.current, version.status = False, "withdrawn"
            revision(s, True)
            event(s, actor, "withdraw", version.id)
            return record(version)

    def list_knowledge(self, kind, limit=50, offset=0):
        with self.db.transaction() as s:
            return [self._knowledge(s, row) for row in s.scalars(select(Knowledge).where(Knowledge.kind == kind)
                        .order_by(Knowledge.created_at, Knowledge.id).limit(limit).offset(offset))]

    def _knowledge(self, s, row):
        evidence = list(s.scalars(select(Evidence).join(KnowledgeEvidence, KnowledgeEvidence.evidence_id == Evidence.id)
                                 .where(KnowledgeEvidence.knowledge_id == row.id)))
        return {**record(row), "fund": s.get(Program, row.program_id).name,
                "evidence": [record(e) for e in evidence]}

    def get_knowledge(self, kind, identifier):
        with self.db.transaction() as s:
            row = require(s, Knowledge, identifier)
            if row.kind != kind:
                raise ServiceError(404, "Knowledge item not found")
            return self._knowledge(s, row)

    def save_knowledge(self, kind, body, identifier=None):
        if kind == "fact" and (body.amount_type is None or body.conditions):
            raise ServiceError(422, "A fact needs amount_type and cannot contain rule conditions")
        if kind == "rule" and (not body.conditions or body.amount_type is not None or body.amount is not None):
            raise ServiceError(422, "A rule needs conditions and cannot contain amounts")
        if body.amount_type == "allocated" and body.fiscal_year_be is None:
            raise ServiceError(422, "Annual allocations require fiscal_year_be")
        if body.amount_type == "ceiling" and body.fiscal_year_be is not None:
            raise ServiceError(422, "Central ceilings have no fiscal_year_be")
        with self.db.transaction() as s:
            s.execute(text("SELECT pg_advisory_xact_lock(7249302)"))
            if identifier:
                row = require(s, Knowledge, identifier, lock=True)
                if row.kind != kind:
                    raise ServiceError(404, "Knowledge item not found")
                if row.status == "approved":
                    raise ServiceError(409, "Approved knowledge cannot be overwritten")
                s.execute(delete(KnowledgeEvidence).where(KnowledgeEvidence.knowledge_id == row.id))
            else:
                row = Knowledge(kind=kind)
            program = named(s, Program, body.fund)
            row.program_id, row.fiscal_year_be = program.id, body.fiscal_year_be
            row.tier_id = named(s, Tier, body.tier).id if body.tier else None
            if body.fiscal_year_be:
                call = s.scalar(select(Call).where(Call.program_id == program.id, Call.fiscal_year_be == body.fiscal_year_be))
                if call is None:
                    call = Call(program_id=program.id, fiscal_year_be=body.fiscal_year_be)
                    s.add(call)
                    s.flush()
                row.call_id = call.id
            else:
                row.call_id = None
            row.amount_type, row.amount, row.status = body.amount_type, body.amount, "draft"
            if identifier is None:
                s.add(row)
            row.legacy_document_gate, row.reviewed_by, row.reviewed_at = False, None, None
            row.payload = body.model_dump(mode="json", exclude={"evidence"})
            # Flush only after all required fields exist.
            s.flush()
            for ref in body.evidence:
                version = require(s, Version, ref.version_id)
                if not 1 <= ref.page <= version.pages_total:
                    raise ServiceError(422, "Evidence page outside document")
                page = s.scalar(select(Page).where(Page.version_id == ref.version_id, Page.number == ref.page))
                chunk = require(s, Chunk, ref.chunk_id) if ref.chunk_id else None
                if chunk and (chunk.version_id != ref.version_id or chunk.page != ref.page):
                    raise ServiceError(422, "Evidence chunk/version/page mismatch")
                source = chunk.text if chunk else page.text if page else ""
                if ref.quote not in source:
                    raise ServiceError(422, "Quote must occur verbatim in the referenced evidence")
                evidence = Evidence(version_id=ref.version_id, page=ref.page, chunk_id=ref.chunk_id,
                                    quote=ref.quote, clause=ref.clause)
                s.add(evidence)
                s.flush()
                s.add(KnowledgeEvidence(knowledge_id=row.id, evidence_id=evidence.id))
            s.flush()
            revision(s, True)
            event(s, "admin", "save_" + kind, row.id)
            return self._knowledge(s, row)

    def decide_knowledge(self, kind, identifier, body):
        with self.db.transaction() as s:
            row = require(s, Knowledge, identifier, lock=True)
            if row.kind != kind:
                raise ServiceError(404, "Knowledge item not found")
            if row.status != "draft":
                raise ServiceError(409, "Knowledge already reviewed")
            refs = list(s.scalars(select(Evidence).join(KnowledgeEvidence, KnowledgeEvidence.evidence_id == Evidence.id)
                                  .where(KnowledgeEvidence.knowledge_id == row.id)))
            if body.decision == "approved":
                if not refs:
                    raise ServiceError(409, "Knowledge requires source evidence")
                for ref in refs:
                    version = require(s, Version, ref.version_id, lock=True)
                    if version.status != "published" or not version.current:
                        raise ServiceError(409, "Evidence must belong to a current published document")
                    chunk = s.get(Chunk, ref.chunk_id) if ref.chunk_id else None
                    page = s.scalar(select(Page).where(Page.version_id == ref.version_id, Page.number == ref.page))
                    if ref.quote not in (chunk.text if chunk else page.text if page else ""):
                        raise ServiceError(409, "Evidence quote no longer matches")
            row.status, row.reviewed_by, row.reviewed_at = body.decision, body.reviewer, now()
            revision(s, True)
            event(s, body.reviewer, body.decision + "_" + kind, row.id)
            return self._knowledge(s, row)

    def jobs(self, limit=50, offset=0):
        with self.db.transaction() as s:
            return [record(j) for j in s.scalars(select(Job).order_by(Job.created_at.desc()).limit(limit).offset(offset))]

    def get_job(self, identifier):
        with self.db.transaction() as s:
            return record(require(s, Job, identifier))

    def retry(self, identifier):
        with self.db.transaction() as s:
            job = require(s, Job, identifier, lock=True)
            if job.status not in {"failed", "needs_ocr"}:
                raise ServiceError(409, "Only failed jobs can be retried")
            version = require(s, Version, job.version_id, lock=True) if job.version_id else None
            if version and (version.status != "draft" or s.scalar(select(Page.id).where(
                    Page.version_id == version.id, Page.reviewed_by.is_not(None)).limit(1))):
                raise ServiceError(409, "Retry cannot overwrite human review")
            if job.payload.get("human_edited"):
                raise ServiceError(409, "Retry cannot overwrite human edits")
            if version:
                if s.scalar(select(Evidence.id).where(Evidence.version_id == version.id).limit(1)):
                    raise ServiceError(409, "Retry cannot replace referenced evidence")
                s.execute(delete(Chunk).where(Chunk.version_id == version.id))
                s.execute(delete(Page).where(Page.version_id == version.id))
            job.status, job.attempts, job.error, job.owner, job.lease_until = "queued", 0, None, None, None
            event(s, "admin", "retry_job", job.id)
            return record(job)

    def claim(self, owner):
        with self.db.transaction() as s:
            # Expired exhausted work is terminal instead of remaining 'running' forever.
            s.execute(update(Job).where(Job.status == "running", Job.lease_until < func.now(),
                      Job.attempts >= self.settings.job_max_attempts).values(status="failed", error="Retry limit exceeded"))
            job = s.scalar(select(Job).where(or_(Job.status == "queued", and_(Job.status == "running", Job.lease_until < func.now())),
                           Job.attempts < self.settings.job_max_attempts)
                           .order_by(Job.created_at).with_for_update(skip_locked=True).limit(1))
            if job is None:
                return None
            job.owner, job.status, job.attempts = owner, "running", job.attempts + 1
            job.heartbeat_at = now()
            job.lease_until = now() + timedelta(seconds=self.settings.job_lease_seconds)
            return record(job)

    def heartbeat(self, identifier, owner):
        with self.db.transaction() as s:
            return bool(s.execute(update(Job).where(Job.id == identifier, Job.owner == owner, Job.status == "running",
                         Job.lease_until > func.now()).values(heartbeat_at=now(),
                         lease_until=now() + timedelta(seconds=self.settings.job_lease_seconds))).rowcount)

    def job_source(self, job):
        with self.db.transaction() as s:
            version = require(s, Version, job["version_id"])
            return self.files.checked(version.file_key, version.sha256)

    def finish(self, identifier, owner, pages=None, error=None):
        with self.db.transaction() as s:
            job = require(s, Job, identifier, lock=True)
            if job.owner != owner or job.status != "running" or job.lease_until <= now():
                return False
            if error:
                job.status = "queued" if job.attempts < self.settings.job_max_attempts else "failed"
                job.error = error[:2000]
            elif job.kind == "ingest":
                version = require(s, Version, job.version_id, lock=True)
                if version.status != "draft":
                    raise ServiceError(409, "Cannot overwrite reviewed text")
                if not pages or len(pages) != version.pages_total:
                    raise ServiceError(409, "Extractor did not return every page")
                if s.scalar(select(Page.id).where(Page.version_id == version.id).limit(1)):
                    raise ServiceError(409, "Refusing to overwrite existing extracted pages")
                chunk_set = s.scalar(select(ChunkSet).where(ChunkSet.version_id == version.id))
                for item in pages:
                    page = Page(version_id=version.id, **item)
                    s.add(page)
                    chunk_page(s, version, page, chunk_set)
                job.status = "needs_ocr" if any(p["error"] for p in pages) else "awaiting_review"
            else:
                job.status = "done"
            job.lease_until, job.owner = None, None
            return True

    def add_feedback(self, body):
        with self.db.transaction() as s:
            require(s, QueryRun, body.request_id)
            row = Feedback(request_id=body.request_id, helpful=body.helpful, note=body.note)
            s.add(row)
            s.flush()
            return record(row)

    def save_query(self, identifier, question, response, model_revision):
        with self.db.transaction() as s:
            s.add(QueryRun(id=identifier, question=question, response=response, model_revision=model_revision))

    def corpus_revision(self):
        with self.db.transaction() as s:
            return revision(s)
