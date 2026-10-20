"""Keep the v1 wire format while storing all new work in PostgreSQL."""
import uuid
from types import SimpleNamespace
from sqlalchemy import select, update
from app.core.errors import ServiceError
from app.modules.knowledge.schemas import Actor, Decision, ReviewDocument
from .models import Chunk, Document, Evidence, Job, Knowledge, Page, Version
from .repository import event, now, require, revision


class LegacyAdapter:
    def __init__(self, services):
        self.services, self.repo, self.db = services, services.repository, services.database

    def resolve(self, s, model, identifier):
        try:
            uid = uuid.UUID(identifier)
        except (ValueError, TypeError, AttributeError):
            uid = None
        row = s.get(model, uid) if uid else None
        if row is None and hasattr(model, "legacy_id"):
            row = s.scalar(select(model).where(model.legacy_id == identifier))
        if row is None:
            raise ServiceError(404, f"{model.__name__} not found")
        return row

    def _version(self, s, identifier):
        doc = self.resolve(s, Document, identifier)
        return s.scalar(select(Version).where(Version.document_id == doc.id).order_by(Version.number.desc()).limit(1))

    @staticmethod
    def job_status(job):
        return "extracting" if job.status == "running" else job.status

    def document(self, s, doc):
        v = self._version(s, str(doc.id))
        job = s.scalar(select(Job).where(Job.version_id == v.id, Job.kind == "ingest").order_by(Job.created_at.desc()))
        chunks = list(s.scalars(select(Chunk).where(Chunk.version_id == v.id)))
        pages = list(s.scalars(select(Page).where(Page.version_id == v.id)))
        return {"document_id": str(doc.id), "job_id": str(job.id) if job else str(v.id), "filename": v.filename,
                "content_sha256": v.sha256, "content_bytes": self.services.files.path(v.file_key).stat().st_size,
                "pages_total": v.pages_total, "pages_processed": len(pages), "chunk_count": len(chunks),
                "status": self.job_status(job) if job and job.status not in {"done"} else "awaiting_review",
                "verified": v.status in {"reviewed", "published"}, "published": v.current,
                "created_at": v.created_at.isoformat(), "updated_at": v.created_at.isoformat(),
                "error": job.error if job else None, "needs_ocr_pages": [p.number for p in pages if p.error]}

    def create_document(self, content, filename):
        result = self.repo.create_document(content, filename)
        with self.db.transaction() as s:
            v = s.get(Version, result["version_id"])
            job = s.get(Job, result["job_id"]) if result["job_id"] else None
            return {"document_id": str(v.document_id), "job_id": str(job.id if job else v.id),
                    "status": self.job_status(job) if job else "awaiting_review", "duplicate": result["duplicate"]}

    def get_document(self, identifier):
        with self.db.transaction() as s:
            return self.document(s, self.resolve(s, Document, identifier))

    def list_documents(self, limit=None, offset=0):
        with self.db.transaction() as s:
            query = select(Document).order_by(Document.created_at).offset(offset)
            if limit:
                query = query.limit(limit)
            return [self.document(s, d) for d in s.scalars(query)]

    def source_path(self, identifier):
        with self.db.transaction() as s:
            v = self._version(s, identifier)
            return self.services.files.path(v.file_key)

    def source_document(self, label):
        with self.db.transaction() as s:
            d = s.scalar(select(Document).where(Document.title == label))
            if not d:
                return None
            return self.services.files.path(self._version(s, str(d.id)).file_key)

    def list_chunks(self, identifier):
        with self.db.transaction() as s:
            v = self._version(s, identifier)
            d = s.get(Document, v.document_id)
            chunks = list(s.scalars(select(Chunk).where(Chunk.version_id == v.id).order_by(Chunk.page, Chunk.id)))
            return [{**c.payload, "id": c.legacy_id or str(c.id), "document_id": str(d.id), "doc": d.title,
                     "fiscal_year": d.fiscal_year_be, "source_filename": v.filename, "text": c.text, "page": c.page,
                     "extraction_method": c.payload.get("extraction_method", "legacy"), "order_in_page": i,
                     "global_order": i, "clause": c.payload.get("clause", ""), "part_index": c.payload.get("part_index", 0),
                     "section_type": c.payload.get("section_type", "other"), "section_label": c.payload.get("section_label", "ข้อความอื่น"),
                     "status": "active" if c.verified else "draft", "verified": c.verified, "published": v.current,
                     "created_at": c.created_at.isoformat(), "updated_at": c.created_at.isoformat()}
                    for i, c in enumerate(chunks)]

    def get_job(self, identifier):
        with self.db.transaction() as s:
            j = self.resolve(s, Job, identifier)
            v = s.get(Version, j.version_id)
            return {"job_id": str(j.id), "document_id": str(v.document_id), "status": self.job_status(j),
                    "attempts": j.attempts, "created_at": j.created_at.isoformat(), "updated_at": j.created_at.isoformat(),
                    "error": j.error, "pages_total": v.pages_total}

    def retry_job(self, identifier):
        with self.db.transaction() as s:
            uid = self.resolve(s, Job, identifier).id
        self.repo.retry(uid)
        return self.get_job(str(uid))

    def update_chunk(self, document_id, chunk_id, reviewer, changes):
        with self.db.transaction() as s:
            v = self._version(s, document_id)
            v = require(s, Version, v.id, lock=True)
            if v.status != "draft":
                raise ServiceError(409, "Published text is immutable")
            c = self.resolve(s, Chunk, chunk_id)
            if c.version_id != v.id:
                raise ServiceError(404, "Chunk not found")
            job = s.scalar(select(Job).where(Job.version_id == v.id, Job.kind == "ingest"))
            if job and job.status != "awaiting_review":
                raise ServiceError(409, "Complete every page before editing")
            if s.scalar(select(Evidence.id).where(Evidence.chunk_id == c.id).limit(1)):
                raise ServiceError(409, "Referenced chunk is immutable; create a new version")
            c.text = changes.get("text", c.text)
            c.payload = {**c.payload, **changes, "last_edited_by": reviewer}
            if job:
                job.payload = {**job.payload, "human_edited": True}
            page = s.scalar(select(Page).where(Page.version_id == v.id, Page.number == c.page))
            if page:
                s.flush()
                page.text = "\n".join(s.scalars(select(Chunk.text).where(Chunk.version_id == v.id, Chunk.page == c.page)))
            event(s, reviewer, "edit_chunk", c.id)
        return next(c for c in self.list_chunks(document_id) if c["id"] == chunk_id)

    def publish(self, document_id, reviewer, source_sha256, confirmed_pages):
        with self.db.transaction() as s:
            v = self._version(s, document_id)
            uid, state = v.id, v.status
        if state == "published":
            return {"document_id": document_id, "published": True, "already_published": True,
                    "chunk_count": len(self.list_chunks(document_id))}
        if state == "draft":
            self.repo.review_document(uid, ReviewDocument(reviewer=reviewer, source_sha256=source_sha256,
                                                          confirmed_pages=confirmed_pages))
        self.repo.publish(uid, reviewer)
        return {"document_id": document_id, "published": True, "already_published": False,
                "chunk_count": len(self.list_chunks(document_id))}

    def verify_fact(self, fact_id, reviewer, amount=None):
        if not reviewer.strip():
            raise ServiceError(422, "Reviewer required")
        with self.db.transaction() as s:
            row = self.resolve(s, Knowledge, fact_id)
            if row.status != "draft":
                return {"ok": False, "code": "already_verified", "error": "Already reviewed"}
            if row.amount is not None and amount is None:
                return {"ok": False, "error": "Confirm the source amount"}
            if amount is not None:
                if type(amount) is not int or amount < 0:
                    return {"ok": False, "error": "Invalid amount"}
                row.amount = amount
            uid = row.id
        result = self.repo.decide_knowledge("fact", uid, Decision(reviewer=reviewer, decision="approved"))
        return {"ok": True, "fact": result}

    def verify_chunks(self, doc, reviewer):
        with self.db.transaction() as s:
            d = s.scalar(select(Document).where(Document.title == doc))
            if d is None:
                raise ServiceError(404, "Document not found")
            v = self._version(s, str(d.id))
            identifier, digest, count = str(d.id), v.sha256, v.pages_total
        self.publish(identifier, reviewer, digest, list(range(1, count + 1)))
        return {"ok": True, "doc": doc, "approved": len(self.list_chunks(identifier))}


    def record_query(self, result, user_id):
        from .models import QueryRun, Feedback
        identifier = uuid.uuid4()
        # Keep the compatibility audit payload free of employment/profile records.
        response = {k: result.get(k) for k in ("answer", "citations", "slots", "ruling", "verify", "timings_ms", "answer_source")}
        response["user_id"] = user_id
        with self.db.transaction() as s:
            s.add(QueryRun(id=identifier, question=result.get("question") or "", response=response))
            if result.get("answer_source") == "feedback":
                s.flush()
                s.add(Feedback(request_id=identifier, helpful=result.get("ruling", {}).get("verdict") == "helpful",
                               note=result.get("slots", {}).get("note", "")))

    def recent(self, limit):
        from .models import QueryRun
        with self.db.transaction() as s:
            rows = list(s.scalars(select(QueryRun).order_by(QueryRun.created_at.desc()).limit(max(0, min(limit, 200)))))
            return [{**r.response, "question": r.question, "at": r.created_at.isoformat(),
                     "verdict": (r.response.get("ruling") or {}).get("verdict"),
                     "blocked": (r.response.get("verify") or {}).get("ok") is False} for r in reversed(rows)]
