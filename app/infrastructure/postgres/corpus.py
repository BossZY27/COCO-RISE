"""Build a detached snapshot from one repeatable-read transaction."""
import threading
from decimal import Decimal
from sqlalchemy import select, text
from app.modules.knowledge import corpus as store
from .models import (Alias, Chunk, CorpusRevision, Document, Evidence, Knowledge,
                     KnowledgeEvidence, Persona, Program, Tier, Version)
from .repository import revision


def amount(value):
    if value is None:
        return None
    number = Decimal(value)
    return int(number) if number == number.to_integral_value() else number


class CorpusRepository:
    def __init__(self, database):
        self.db = database
        self._cache = None
        self._lock = threading.Lock()

    def load(self):
        with self.db.transaction() as s:
            s.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
            rev = revision(s)
            with self._lock:
                if self._cache is not None and self._cache.revision == rev:
                    return self._cache
            documents = {d.id: d for d in s.scalars(select(Document))}
            versions = {v.id: v for v in s.scalars(select(Version))}
            programs = {p.id: p.name for p in s.scalars(select(Program))}
            tiers = {t.id: t.name for t in s.scalars(select(Tier))}
            chunks = []
            chunk_rows = {c.id: c for c in s.scalars(select(Chunk))}
            active_versions = {v.id for v in versions.values() if v.current and v.status == "published"}
            for row in chunk_rows.values():
                version = versions[row.version_id]
                doc = documents[version.document_id]
                chunks.append({**row.payload, "id": row.legacy_id or str(row.id), "text": row.text,
                               "doc": doc.title, "fiscal_year": doc.fiscal_year_be, "page": row.page,
                               "clause": row.payload.get("clause", ""),
                               "status": "active" if row.version_id in active_versions and row.verified else "draft",
                               "verified": row.verified and row.version_id in active_versions,
                               "document_id": str(doc.id), "version_id": str(version.id), "chunk_id": str(row.id)})
            evidence = {e.id: e for e in s.scalars(select(Evidence))}
            links = {}
            for link in s.scalars(select(KnowledgeEvidence)):
                links.setdefault(link.knowledge_id, []).append(evidence[link.evidence_id])
            facts, rules, citations = [], [], {}
            for row in s.scalars(select(Knowledge)):
                refs = links.get(row.id, [])
                live = bool(refs) and all(e.version_id in active_versions for e in refs)
                legacy_gate = row.legacy_document_gate and live and all(
                    c.verified for c in chunk_rows.values() if c.version_id in {e.version_id for e in refs})
                verified = live and (row.status == "approved" or legacy_gate)
                source = dict(row.payload.get("source") or {})
                if refs:
                    ref = refs[0]
                    version = versions[ref.version_id]
                    source = {"doc": documents[version.document_id].title, "page": ref.page, "clause": ref.clause}
                payload = {**row.payload, "id": row.legacy_id or str(row.id), "fund": programs[row.program_id],
                           "fiscal_year": row.fiscal_year_be, "source": source, "verified": verified}
                if row.kind == "fact":
                    payload.update(amount_type=row.amount_type, amount=amount(row.amount), tier=tiers.get(row.tier_id),
                                   total_cap=amount(payload.get("total_cap")),
                                   max_grants_per_person=payload.get("max_grants_per_person"),
                                   max_grants_per_year=payload.get("max_grants_per_year"))
                    facts.append(payload)
                elif verified:
                    rules.append(payload)
                citations[payload["id"]] = [self.citation(e, versions, documents) for e in refs]
            aliases = {"aliases": {}, "ambiguous": {}}
            for a in s.scalars(select(Alias)):
                if len(a.programs) == 1:
                    aliases["aliases"].setdefault(a.programs[0], []).append(a.name)
                else:
                    aliases["ambiguous"][a.name] = a.programs
            personas = {p.legacy_id: p.payload for p in s.scalars(select(Persona))}
        corpus = store.make_corpus(facts, chunks, rules, personas, aliases, rev)
        corpus.citations = citations
        with self._lock:
            self._cache = corpus
        return corpus

    @staticmethod
    def citation(e, versions, documents):
        v = versions[e.version_id]
        return {"document_id": str(v.document_id), "version_id": str(v.id), "page": e.page,
                "chunk_id": str(e.chunk_id) if e.chunk_id else None, "quote": e.quote,
                "label": documents[v.document_id].title, "clause": e.clause,
                "url": f"/v2/sources/{v.id}/pages/{e.page}"}
