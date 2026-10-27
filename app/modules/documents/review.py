"""The human review step from the architecture, as working code.

A script can read a number off a page. It cannot take responsibility for it. So every number and every
OCR'd passage starts life unverified, stays out of the answer path, and becomes
trusted only when a person opens the source document and says so here.

Writes go through a temporary file and a replace, so a crash midway leaves the old file intact.
"""
import json
import os
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path

from app.modules.knowledge import corpus as store

DATA = Path(__file__).resolve().parents[3] / "data"
SOURCE_DIR = Path(os.getenv(
    "RISE_SOURCE_DIR",
    str(Path(__file__).resolve().parents[5] / "RISE" / "resource" / "Docs"
        / "Seminar_resource" / "Funds_resource"),
))
TOKEN = os.getenv("RISE_ADMIN_TOKEN", "")
_LOCK = threading.Lock()
_DOC_NUMBER = re.compile(r"(\d{1,2})[-_/](\d{2,4})")


def enabled() -> bool:
    """No token, no admin surface. A review tool that anyone can reach is worse than none."""
    return bool(TOKEN)


def authorised(supplied: str | None) -> bool:
    return enabled() and supplied == TOKEN


def _write(name: str, rows: list) -> None:
    path = DATA / name
    backup = DATA / f"{name}.bak"
    temp = DATA / f"{name}.tmp"
    temp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    if path.exists():
        shutil.copy2(path, backup)
    temp.replace(path)


def pending() -> dict:
    """What still needs a person: money rows nobody has checked, and passages OCR guessed at."""
    facts = [f for f in store.view().FACTS if not store.is_verified(f)]
    chunks = [c for c in store.view().CHUNKS if not store.is_verified(c)]
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c["doc"]] = by_doc.get(c["doc"], 0) + 1
    return {"facts": facts, "chunk_count": len(chunks), "chunks_by_doc": by_doc,
            "chunks": chunks, "admin_enabled": enabled()}


def _source_label(path: Path) -> str | None:
    if "หลักเกณฑ์" in path.stem:
        return "หลักเกณฑ์กลาง"
    match = _DOC_NUMBER.search(path.stem)
    if match is None:
        return None
    year = match.group(2)
    return f"ประกาศ {match.group(1)}/{('25' + year) if len(year) == 2 else year}"


def source_document(doc: str) -> Path | None:
    """Resolve only a known grant document to a PDF under the configured source directory."""
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.source_document(doc)
    known = {c["doc"] for c in store.view().CHUNKS}
    known.update(f["source"]["doc"] for f in store.view().FACTS)
    if doc not in known or not SOURCE_DIR.is_absolute():
        return None
    root = SOURCE_DIR.resolve()
    matches = [path.resolve() for path in root.glob("*.pdf")
               if _source_label(path) == doc and path.resolve().is_relative_to(root)
               and path.is_file()]
    return matches[0] if len(matches) == 1 else None


def verify_fact(fact_id: str, reviewer: str, amount: int | None = None) -> dict:
    """Mark one amount as checked against the source, correcting the figure when the reviewer saw a different one."""
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.verify_fact(fact_id, reviewer, amount)
    reviewer = reviewer.strip() if isinstance(reviewer, str) else ""
    if not reviewer:
        return {"ok": False, "error": "กรุณาระบุชื่อผู้ตรวจที่เปิดเอกสารต้นฉบับ"}
    if amount is not None and (type(amount) is not int or amount < 0):
        return {"ok": False, "error": "จำนวนเงินต้องเป็นเลขจำนวนเต็มที่ไม่ติดลบ"}
    with _LOCK:
        rows = json.loads((DATA / "facts.json").read_text(encoding="utf-8"))
        target = next((r for r in rows if r["id"] == fact_id), None)
        if target is None:
            return {"ok": False, "error": f"ไม่พบข้อเท็จจริง {fact_id}"}
        if store.is_verified(target):
            return {"ok": False, "code": "already_verified",
                    "error": "ข้อเท็จจริงนี้อนุมัติแล้ว จึงแก้ทับผ่านขั้นตอนอนุมัติไม่ได้"}
        if target["amount"] is not None and amount is None:
            return {"ok": False, "error": "กรุณายืนยันตัวเลขจากเอกสารต้นฉบับก่อนอนุมัติ"}
        if amount is not None and amount != target["amount"]:
            target["amount"] = amount
        target["verified"] = True
        target["reviewed_by"] = reviewer
        target["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
        _write("facts.json", rows)
        for live in store.view().FACTS:          # the running server must see it without a restart
            if live["id"] == fact_id:
                live.update(target)
        store.refresh_derived_indexes()
        return {"ok": True, "fact": target}


def verify_chunks(doc: str, reviewer: str) -> dict:
    """Approve the OCR and hand-transcribed passages only after a person reviews the document."""
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.verify_chunks(doc, reviewer)
    reviewer = reviewer.strip() if isinstance(reviewer, str) else ""
    if not reviewer:
        return {"ok": False, "error": "กรุณาระบุชื่อผู้ตรวจที่เปิดเอกสารต้นฉบับ"}
    with _LOCK:
        files = {}
        touched = 0
        for name in ("chunks.json", "chunks_ingested.json"):
            path = DATA / name
            if not path.exists():
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))
            files[name] = rows
            for row in rows:
                if row["doc"] == doc and not store.is_verified(row):
                    row.update({"verified": True, "status": "active", "reviewed_by": reviewer,
                                "reviewed_at": datetime.now().isoformat(timespec="seconds")})
                    touched += 1
        if not touched:
            return {"ok": False, "error": f"ไม่มีชิ้นข้อความของ {doc} ที่รอตรวจ"}
        for name, rows in files.items():
            _write(name, rows)
        approved = {r["id"]: r for rows in files.values() for r in rows}
        for live in store.view().CHUNKS:
            if live["id"] in approved:
                live.update(approved[live["id"]])
        store.refresh_derived_indexes()
        return {"ok": True, "doc": doc, "approved": touched,
                "note": "ชิ้นที่อนุมัติจะถูกค้นเจอหลังสร้างเวกเตอร์ใหม่ด้วย python reindex.py"}
