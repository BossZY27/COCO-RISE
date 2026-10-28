"""Fail-closed answer verification, independent of API and query orchestration."""

import re
import json

from app.modules.knowledge import corpus as store
from app.modules.knowledge.formatting import baht as _baht, cite as _cite

NUMBER_RE = re.compile(r"\d[\d,]*")
REASONING_PREAMBLE = re.compile(r"^\s*(?:okay[, ]|alright[, ]|let's\b|let me\b|the user\b|first[, ]|i need\b|i should\b)", re.I)
QUALITATIVE_REQUIREMENT = re.compile(
    r"(?:ผู้สมัคร|ผู้ขอรับทุน|ผู้ขอ).{0,80}?(?:ต้อง(?:มี|เป็น|สังกัด|เคย|ผ่าน)|ห้าม|จำเป็น|มีสิทธิ์?|ไม่มีสิทธิ์?)"
)


def verify(answer: str, citations: list[str], used_facts: list[dict], evidence: dict, slots: dict,
           check_numbers: bool = True, ruling: dict | None = None) -> dict:
    """The number gate exists to catch a model inventing figures. An answer the code assembled from
    verified rows, such as the fund list, has no model in its path, so it is checked for citations only."""
    blob = " ".join(c["text"] for c in evidence["chunks"]) + " " + " ".join(citations)
    for fact in used_facts:
        blob += f" {_baht(fact['amount'])} {fact['amount']}" if fact["amount"] else ""
        blob += f" {_baht(fact['total_cap'])}" if fact["total_cap"] else ""
        blob += f" {fact['max_grants_per_person']}" if fact["max_grants_per_person"] else ""
        blob += f" {fact.get('max_grants_per_year')}" if fact.get("max_grants_per_year") else ""
        blob += f" {fact['fiscal_year']}" if fact["fiscal_year"] else ""
        blob += f" {fact.get('note') or ''}"
    blob += f" {(ruling or {}).get('reason') or ''}"          # a reviewed rule quoted in the answer
    for rule in evidence.get("rules", []):
        blob += " " + json.dumps(rule.get("conditions", []), ensure_ascii=False)
    problems = []
    if any(not store.is_verified(r) for r in evidence.get("rules", [])):
        problems.append("Rule has not been reviewed")
    if REASONING_PREAMBLE.search(answer):
        problems.append("คำตอบเป็นข้อความคิดของโมเดล ไม่ใช่คำตอบภาษาไทยสำหรับผู้ใช้")
    qualitative_support = " ".join(
        [str(fact.get("note") or "") for fact in used_facts]
        + [str((ruling or {}).get("reason") or "")]
    )
    for claim in (QUALITATIVE_REQUIREMENT.findall(answer) if used_facts else []):
        if claim not in qualitative_support:
            problems.append("คำตอบเพิ่มเงื่อนไขเชิงคุณสมบัติที่ไม่มีในข้อเท็จจริงหรือกฎที่ตรวจแล้ว")
            break
    allowed_numbers = {
        number.strip(",").replace(",", "")
        for number in NUMBER_RE.findall(blob)
    }
    for number in (NUMBER_RE.findall(answer) if check_numbers else []):
        if number == str(slots["fiscal_year"]):   # the year comes from the question, not from a document
            continue
        if number.strip(",").replace(",", "") not in allowed_numbers:
            problems.append(f"ตัวเลข {number} ไม่พบในหลักฐาน")
    if used_facts and slots["fiscal_year"] and used_facts[0]["amount_type"] == "allocated":
        if used_facts[0]["fiscal_year"] != slots["fiscal_year"]:
            problems.append("ปีงบประมาณของหลักฐานไม่ตรงกับคำถาม")
    if used_facts and not citations:
        problems.append("คำตอบไม่มีการอ้างอิง")
    for fact in used_facts:
        source = fact.get("source")
        if source:
            # Older API clients omit the page but still identify the exact document
            # and clause.  That is enough to establish coverage; the UI normally
            # receives the fuller citation produced by text.cite().
            covered = any(source.get("doc", "") in citation and source.get("clause", "") in citation
                          for citation in citations)
            if not covered:
                problems.append(f"คำตอบไม่มีการอ้างอิงหลักฐานของ {fact.get('fund') or 'ข้อเท็จจริง'}")
    # Right digits attached to the wrong fund reads as a confident, cited, wrong answer. The number
    # gate passes it, because the figure really is in the evidence, just under another heading.
    if used_facts and used_facts[0].get("amount") and check_numbers:
        figure = f"{used_facts[0]['amount']:,}"
        if figure not in answer and str(used_facts[0]["amount"]) not in answer:
            problems.append(f"คำตอบไม่ได้บอกจำนวนเงิน {figure} ที่หลักฐานระบุไว้")
        if used_facts[0]["amount_type"] == "allocated" and not any(
                word in answer for word in ("สูงสุด", "ไม่เกิน")):
            problems.append("วงเงินจัดสรรเป็นวงเงินสูงสุด ไม่ใช่จำนวนที่จะได้รับแน่นอน")
        if used_facts[0]["amount_type"] == "ceiling" and not any(
                word in answer for word in ("เพดาน", "สูงสุด", "ไม่เกิน")):
            problems.append("ตัวเลขเพดานต้องระบุว่าเป็นเพดาน ไม่ใช่จำนวนที่จะได้รับแน่นอน")
        fund = used_facts[0].get("fund")
        if fund and not any(term in answer for term in (fund, "ทุนนี้", "ทุนนั้น")):
            problems.append("คำตอบไม่ได้ระบุว่าตัวเลขเป็นของทุนใด")
    if len(used_facts) > 1 and check_numbers:
        for fact in used_facts:
            amount = fact.get("amount")
            if amount and f"{amount:,}" not in answer and str(amount) not in answer:
                problems.append(f"คำตอบตัดจำนวนเงิน {amount:,} ของหลักฐานบางแถวออก")
            tier = fact.get("tier")
            if tier and tier not in answer:
                problems.append(f"คำตอบตัดข้อมูล{tier}ออก")
            fund = fact.get("fund")
            if fund and fund not in answer:
                problems.append(f"คำตอบตัดข้อมูลของ{fund}ออก")
    # A rate note can hold the condition that makes the figure true, such as "ร้อยละ 10". A draft that keeps
    # the figure and drops the condition passes every check above. Whole numbers are compared, because
    # "10" is a substring of "100,000".
    if used_facts and check_numbers and used_facts[0].get("note"):
        said = {n.strip(",").replace(",", "") for n in NUMBER_RE.findall(answer)}
        dropped = [n.strip(",") for n in NUMBER_RE.findall(used_facts[0]["note"])
                   if n.strip(",").replace(",", "") not in said]
        if dropped:
            problems.append(f"คำตอบตัดเงื่อนไขในหมายเหตุอัตราออก ({dropped[0]})")
    if any(not store.is_verified(fact) for fact in used_facts):
        problems.append("ข้อมูลจากเอกสารสแกนยังไม่ผ่านการอนุมัติ")
    if evidence.get("ceiling_fallback") and used_facts and not all(
            word in answer for word in ("เพดาน", "ยังไม่มีข้อมูลวงเงินจัดสรร")):
        problems.append("คำตอบไม่ได้แยกเพดานกลางจากวงเงินจัดสรรรายปีที่ยังไม่มี")
    if (evidence.get("pending_review") and used_facts and
            used_facts[0]["amount_type"] == "ceiling" and slots.get("intent") == "amount" and
            not all(word in answer for word in ("รอ", "อนุมัติ", "เพดาน", "ไม่ใช่วงเงินจัดสรร"))):
        problems.append("คำตอบไม่ได้แยกเพดานกลางจากประกาศจัดสรรที่ยังรออนุมัติ")
    if used_facts:
        # a comparison quotes several funds, and an ambiguous name says which other fund it could mean
        answered_about = {fact.get("fund") for fact in used_facts} | set(slots.get("alternatives") or [])
        if slots.get("intent") == "catalogue_details":
            answered_about.update(row["fund"] for row in store.fund_catalogue()
                                  if _cite(row["source"]) in citations)
        allowed_blockers = set()
        if (ruling and ruling.get("verdict") == "not_eligible" and ruling.get("source")
                and _cite(ruling["source"]) in citations):
            allowed_blockers = {f for f in store.view().FUNDS if f in (ruling.get("reason") or "")}
        # A reviewed rate note may name a related fund ("อัตราเดียวกับทุนนำเสนอผลงานทางวิชาการ"). Quoted
        # whole it is the document speaking, so it is set aside; the same name anywhere else is a stray.
        scanned = answer
        for fact in used_facts:
            if fact.get("note"):
                scanned = scanned.replace(fact["note"], " ")
        strays = [f for f in store.view().FUNDS if f not in answered_about and f not in allowed_blockers and f in scanned]
        if strays:
            problems.append(f"คำตอบกล่าวถึง {strays[0]} ซึ่งไม่ใช่ทุนที่หลักฐานพูดถึง")
    return {"ok": not problems, "problems": problems}
