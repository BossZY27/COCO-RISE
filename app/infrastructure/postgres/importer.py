"""Idempotent insert-only seed import. Existing reviewer work always wins."""
import hashlib
import json
import re
from pathlib import Path
from sqlalchemy import select, text
from .models import (Alias, Call, Chunk, ChunkSet, Document, Evidence, ImportRun, Job,
                     Knowledge, KnowledgeEvidence, Page, Persona, Program, Tier, Version)
from .repository import named, revision


def source_label(path):
    if "หลักเกณฑ์" in path.stem:
        return "หลักเกณฑ์กลาง"
    match = re.search(r"(\d{1,2})[-_/](\d{2,4})", path.stem)
    if match:
        year = match[2] if len(match[2]) == 4 else "25" + match[2]
        return f"ประกาศ {match[1]}/{year}"


def import_legacy(database, files, data_dir: Path, source_dir: Path):
    manifest = {}
    inputs = {}
    for name in ("facts.json", "rules.json", "chunks.json", "chunks_ingested.json",
                 "chunks_published.json", "personas.json", "aliases.json"):
        path = data_dir / name
        if path.exists():
            content = path.read_bytes()
            manifest[name] = hashlib.sha256(content).hexdigest()
            inputs[name] = json.loads(content)
    if any(name not in inputs for name in ("facts.json", "rules.json", "chunks.json", "personas.json", "aliases.json")):
        raise ValueError("Missing required legacy fixtures")
    pdfs = {}
    for path in sorted(source_dir.glob("*.pdf")):
        label = source_label(path)
        if label:
            content = path.read_bytes()
            pages = files.validate(content)
            digest, key = files.save(content)
            manifest["pdf/" + path.name] = digest
            pdfs[label] = (path.name, digest, key, pages)
    all_chunks = inputs["chunks.json"] + inputs.get("chunks_ingested.json", [])
    labels = {c["doc"] for c in all_chunks} | {f["source"]["doc"] for f in inputs["facts.json"]}
    missing = labels - pdfs.keys()
    if missing:
        raise ValueError(f"Missing original PDFs for: {sorted(missing)}")
    # Include persisted quarantine records and PDFs, without inventing a completed job.
    quarantine = data_dir / "ingestion"
    quarantine_records = []
    for path in sorted(quarantine.rglob("document.json")):
        content = path.read_bytes()
        manifest[str(path.relative_to(data_dir))] = hashlib.sha256(content).hexdigest()
        quarantine_records.append((path.parent, json.loads(content)))
    for path in sorted(quarantine.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".pdf"}:
            manifest[str(path.relative_to(data_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    published_documents = {c.get("document_id") for c in inputs.get("chunks_published.json", [])}
    known_quarantine = {v.get("document_id") for _, v in quarantine_records}
    if published_documents - known_quarantine:
        raise ValueError("Published upload records require the original quarantine tree")
    checksum = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    with database.transaction() as s:
        s.execute(text("SELECT pg_advisory_xact_lock(7249303)"))
        versions, chunk_sets = {}, {}
        added = 0
        for label in sorted(labels):
            doc = s.scalar(select(Document).where(Document.legacy_id == label))
            if doc is None:
                doc_chunks = [c for c in all_chunks if c["doc"] == label]
                year = next((c.get("fiscal_year") for c in doc_chunks if c.get("fiscal_year")), None)
                doc = Document(legacy_id=label, title=label, fiscal_year_be=year)
                s.add(doc)
                s.flush()
                filename, digest, key, page_count = pdfs[label]
                reviewed = bool(doc_chunks) and all(c.get("verified") is True and c.get("status") == "active" for c in doc_chunks)
                v = Version(document_id=doc.id, number=1, filename=filename, sha256=digest, file_key=key,
                            pages_total=page_count, current=reviewed, status="published" if reviewed else "draft",
                            provenance={"import": "legacy", "approval": "preserved, no new attestation"})
                s.add(v)
                s.flush()
                cs = ChunkSet(version_id=v.id, strategy="legacy-preserved")
                s.add(cs)
                s.flush()
                added += 1
            else:
                v = s.scalar(select(Version).where(Version.document_id == doc.id).order_by(Version.number))
                cs = s.scalar(select(ChunkSet).where(ChunkSet.version_id == v.id))
            versions[label], chunk_sets[label] = v, cs
        for c in all_chunks:
            if s.scalar(select(Chunk.id).where(Chunk.legacy_id == c["id"])):
                continue
            v = versions[c["doc"]]
            s.add(Chunk(legacy_id=c["id"], version_id=v.id, chunk_set_id=chunk_sets[c["doc"]].id,
                        page=c["page"], text=c["text"], verified=c.get("verified") is True, payload=c))
            added += 1
        s.flush()
        for label, v in versions.items():
            for number in range(1, v.pages_total + 1):
                if s.scalar(select(Page.id).where(Page.version_id == v.id, Page.number == number)):
                    continue
                parts = [c for c in all_chunks if c["doc"] == label and c["page"] == number]
                s.add(Page(version_id=v.id, number=number, text="\n".join(c["text"] for c in parts), method="legacy",
                           error=None if parts else "No imported page text"))
        for kind, filename in (("fact", "facts.json"), ("rule", "rules.json")):
            for value in inputs[filename]:
                if s.scalar(select(Knowledge.id).where(Knowledge.legacy_id == value["id"])):
                    continue
                program = named(s, Program, value["fund"])
                tier = named(s, Tier, value["tier"]) if value.get("tier") else None
                year = value.get("fiscal_year")
                call = None
                if year:
                    call = s.scalar(select(Call).where(Call.program_id == program.id, Call.fiscal_year_be == year))
                    if call is None:
                        call = Call(program_id=program.id, fiscal_year_be=year)
                        s.add(call)
                        s.flush()
                item = Knowledge(legacy_id=value["id"], kind=kind, program_id=program.id, call_id=call.id if call else None,
                                 tier_id=tier.id if tier else None, fiscal_year_be=year,
                                 amount_type=value.get("amount_type"), amount=value.get("amount"),
                                 status="approved" if value.get("verified") is True else "draft",
                                 legacy_document_gate=kind == "rule", payload=value,
                                 reviewed_by=value.get("reviewed_by"), reviewed_at=value.get("reviewed_at"))
                s.add(item)
                s.flush()
                source = value["source"]
                v = versions[source["doc"]]
                candidates = list(s.scalars(select(Chunk).where(Chunk.version_id == v.id, Chunk.page == source["page"])
                                           .order_by(Chunk.legacy_id)))
                clause = source.get("clause", "").replace("บัญชีแนบท้าย ข้อ ", "").replace("ข้อ ", "")
                chunk = next((c for c in candidates if c.payload.get("clause") == clause), None)
                if chunk is None:
                    chunk = next((c for c in candidates if value["fund"] in c.text), candidates[0] if candidates else None)
                evidence = Evidence(version_id=v.id, page=source["page"], chunk_id=chunk.id if chunk else None,
                                    quote=chunk.text if chunk else "", clause=source.get("clause", ""))
                s.add(evidence)
                s.flush()
                s.add(KnowledgeEvidence(knowledge_id=item.id, evidence_id=evidence.id))
                added += 1
        for value in inputs["personas.json"]:
            if not s.scalar(select(Persona.id).where(Persona.legacy_id == value["id"])):
                s.add(Persona(legacy_id=value["id"], payload=value))
                added += 1
        alias_rows = {}
        for fund, aliases in inputs["aliases.json"]["aliases"].items():
            alias_rows.update({a: [fund] for a in aliases})
        alias_rows.update(inputs["aliases.json"]["ambiguous"])
        for alias, programs in alias_rows.items():
            if not s.scalar(select(Alias.id).where(Alias.name == alias)):
                s.add(Alias(name=alias, programs=programs))
                added += 1
        for directory, value in quarantine_records:
            added += import_quarantine(s, files, directory, value, quarantine)
        if added:
            revision(s, True)
            s.add(Job(kind="index"))
        s.add(ImportRun(checksum=checksum, manifest=manifest))
    return {"checksum": checksum, "inserted": added, "files": len(manifest)}


def import_quarantine(s, files, directory, value, quarantine):
    legacy = value["document_id"]
    if s.scalar(select(Document.id).where(Document.legacy_id == legacy)):
        return 0
    source = directory / "source.pdf"
    if not source.exists():
        source = next(iter(directory.glob("*.pdf")), None)
    if source is None:
        raise ValueError(f"Missing quarantine PDF: {legacy}")
    content = source.read_bytes()
    pages = files.validate(content)
    digest, key = files.save(content)
    if value.get("content_sha256") != digest:
        raise ValueError(f"Quarantine PDF checksum mismatch: {legacy}")
    doc = Document(legacy_id=legacy, title=value.get("filename", legacy))
    s.add(doc)
    s.flush()
    published = value.get("published") is True and value.get("verified") is True
    v = Version(document_id=doc.id, number=1, sha256=digest, file_key=key, filename=value["filename"],
                pages_total=pages, status="published" if published else "draft", current=published, provenance=value)
    s.add(v)
    s.flush()
    cs = ChunkSet(version_id=v.id)
    s.add(cs)
    s.flush()
    chunks_path = directory / "chunks.json"
    chunks = json.loads(chunks_path.read_text()) if chunks_path.exists() else []
    for c in chunks:
        if not s.scalar(select(Chunk.id).where(Chunk.legacy_id == c["id"])):
            s.add(Chunk(legacy_id=c["id"], version_id=v.id, chunk_set_id=cs.id, page=c["page"], text=c["text"],
                        verified=c.get("verified") is True, payload=c))
    for number in range(1, pages + 1):
        parts = [c["text"] for c in chunks if c["page"] == number]
        if parts:
            s.add(Page(version_id=v.id, number=number, text="\n".join(parts), method="legacy"))
    job_path = quarantine / "jobs" / (value.get("job_id", "missing") + ".json")
    job = json.loads(job_path.read_text()) if job_path.exists() else {}
    human_edited = bool(value.get("last_edited_at")) or any(c.get("last_edited_at") or c.get("edit_history") for c in chunks)
    state = job.get("status", value.get("status", "queued"))
    if published or human_edited:
        state = "awaiting_review"
    elif state not in {"failed", "needs_ocr", "awaiting_review"}:
        state = "queued"
        # Machine-only partial outputs can be regenerated atomically after restart.
        s.flush()
        from sqlalchemy import delete
        s.execute(delete(Chunk).where(Chunk.version_id == v.id))
        s.execute(delete(Page).where(Page.version_id == v.id))
    s.add(Job(legacy_id=value.get("job_id"), version_id=v.id, status=state,
              attempts=job.get("attempts", 0), error=job.get("error"),
              payload={"legacy": job or value, "human_edited": human_edited}))
    return 1
