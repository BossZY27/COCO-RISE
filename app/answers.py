"""Answers the code builds from reviewed rows when one fund and one figure is not the whole question.

A comparison, a ranking, a list of tiers, the conditions of a fund, who may apply: each is assembled
here from rows a person has verified, with every figure copied from a row and every row cited. No model
touches these answers, because each one names several funds or figures and a rephrasing that dropped one
would still read as complete. pipeline.verify checks them like any other answer.
"""
from . import store, understand
from .text import baht, cite

TIER_ORDER = {"ชั้นต้น": 0, "ชั้นกลาง": 1, "ชั้นสูง": 2}
CEILING_IS_NOT_ALLOCATION = "ตัวเลขเหล่านี้เป็นเพดานของหลักเกณฑ์กลาง ไม่ใช่วงเงินที่จัดสรรจริงของปีงบประมาณใดปีหนึ่ง"
COLLEGE = "วิทยาลัยการคอมพิวเตอร์"
STAFF_TYPES = ("ข้าราชการ", "พนักงานมหาวิทยาลัย", "ลูกจ้างมหาวิทยาลัย")
TOPIC_LIST_MAX = 6          # past this many funds a subject no longer narrows the list
ASK_BY_NAME = "ถามต่อด้วยชื่อทุนที่สนใจ ระบบจะตอบวงเงินและเงื่อนไขของทุนนั้นให้ตรงตัว"
NOT_FOUND = ("ไม่พบทุนที่ระบุเรื่อง{subject}โดยตรงในเอกสารที่ระบบมี ถามว่า มีทุนอะไรบ้าง เพื่อดูรายการทั้งหมด "
             "หรือระบุชื่อทุนที่สนใจ แล้วระบบจะตอบเงื่อนไขของทุนนั้น")
# Answers that already name every fund in play, or name none, need no note about the name asked
NAME_NOTE_SKIP = ("tiers", "compare", "hypothetical", "rank", "list", "topic", "smalltalk")
SMALL_TALK = {
    "greeting": ("สวัสดี ระบบนี้ตอบคำถามเรื่องทุนของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น "
                 "จากเอกสารที่เจ้าหน้าที่ตรวจแล้ว ลองถามเช่น มีทุนอะไรบ้าง หรือ ทุนวิจัยสถาบันได้เท่าไร"),
    "thanks": "ยินดีช่วย ถ้ามีคำถามเรื่องทุนเพิ่มเติม ถามต่อได้เลย",
    "capability": ("ระบบนี้ตอบคำถามเรื่องทุนของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น ได้แก่ มีทุนอะไรบ้าง "
                   "เพดานและวงเงินรวมของแต่ละทุน นิยามในประกาศ ใครขอทุนได้ และสิทธิ์ของคุณตามข้อมูลที่ระบบมี "
                   "ทุกคำตอบอ้างอิงเอกสารที่เจ้าหน้าที่ตรวจแล้ว ส่วนที่ยังรอตรวจ ระบบจะบอกว่ายังรอ"),
}


def ceilings(fund: str) -> list[dict]:
    """The fund's reviewed central-criteria rows: the whole-fund row first, then tiers in order."""
    rows = [f for f in store.fact_lookup(fund, None, None, "ceiling") if store.is_verified(f)]
    return sorted(rows, key=lambda f: -1 if not f["tier"] else TIER_ORDER.get(f["tier"], 9))


def ceiling_terms(row: dict, per_grant: bool = True) -> list[str]:
    """One row's figures in reading order: per grant, grants a year, yearly total, then its rate note.
    per_grant=False leaves out the first when the sentence has already said it its own way."""
    terms = []
    if row["amount"] and per_grant:
        terms.append(f"ไม่เกิน {baht(row['amount'])} บาทต่อทุน")
    if row.get("max_grants_per_year"):
        terms.append(f"ไม่เกิน {row['max_grants_per_year']} ทุนต่อปี")
    if row["total_cap"]:
        terms.append(f"วงเงินรวมไม่เกิน {baht(row['total_cap'])} บาท")
    if row.get("note"):
        terms.append(row["note"])
    return terms


def fund_summary(fund: str) -> tuple[str, list[dict]]:
    """The fund's ceiling in one sentence, and the rows it quotes. Tiers are listed when the fund has them,
    because a tiered fund's whole-fund row carries only the yearly total and read alone it looks untiered."""
    rows = ceilings(fund)
    if not rows:
        return "", []
    tiers = [r for r in rows if r["tier"] and r["amount"]]
    whole = next((r for r in rows if not r["tier"]), None)
    if tiers:
        listed = " ".join(f"{t['tier']} {baht(t['amount'])}" for t in tiers)
        text = f"{fund}มีเพดานแยกตามชั้น {listed} บาทต่อทุน"
        cap = (whole or tiers[0])["total_cap"]
        if cap:
            text += f" วงเงินรวมไม่เกิน {baht(cap)} บาท"
        return text, tiers + ([whole] if whole else [])
    row = whole or rows[0]
    lead = f"{fund}มีเพดาน " if row["amount"] else f"{fund}ไม่ได้กำหนดวงเงินต่อทุน "
    return lead + " ".join(ceiling_terms(row)), [row]


def catalogue_details_answer() -> tuple[str, list[str], list[dict]]:
    """Expand every reviewed catalogue entry without inventing a grant's purpose or open status."""
    sections, used = [], []
    catalogue = store.fund_catalogue()
    for group in ("ทุนส่งเสริมการวิจัย", "ทุนอุดหนุนกิจกรรม"):
        lines = [group]
        for item in catalogue:
            if item["group"] != group:
                continue
            summary, rows = fund_summary(item["fund"])
            if not rows:
                continue
            lines.append(f"ข้อ {item['clause']} {summary}")
            used.extend(rows)
        if len(lines) > 1:
            sections.append("\n".join(lines))
    if not used:
        return "ยังไม่มีรายละเอียดทุนที่ผ่านการตรวจยืนยัน", [], []
    answer = (
        "ข้อมูลรายทุนด้านล่างเป็นชื่อทุนและเพดานตามหลักเกณฑ์กลาง ไม่ใช่วงเงินจัดสรรรายปี "
        "เอกสารที่ตรวจแล้วไม่ได้อธิบายหน้าที่ของทุกทุนเป็นรายข้อ จึงไม่ตีความวัตถุประสงค์จากชื่อทุนเอง\n\n"
        + "\n\n".join(sections)
        + "\n\nข้อมูลนี้ยังไม่ใช่การยืนยันสิทธิ์หรือสถานะเปิดรับ "
        "ถ้าสนใจทุนใด ให้ระบุชื่อทุนเพื่อดูเงื่อนไขที่เอกสารระบุไว้"
    )
    return answer, _unique([cite(row["source"]) for row in used]), used


def _unique(items: list) -> list:
    return list(dict.fromkeys(items))


def tiers_answer(funds: list[str]) -> tuple[str, list[str], list[dict]] | None:
    """"ทุนความเป็นเลิศมีกี่แบบ", "แต่ละชั้นได้เท่าไร": every fund named, with its tiers."""
    parts, used = [], []
    for fund in funds:
        text, rows = fund_summary(fund)
        if rows:
            parts.append(text)
            used += rows
    if not parts:
        return None
    lead = "ตามหลักเกณฑ์กลาง " if len(parts) == 1 else f"คำนี้ตรงกับ {len(parts)} ทุน ตามหลักเกณฑ์กลาง "
    answer = lead + " | ".join(parts) + " " + CEILING_IS_NOT_ALLOCATION
    return answer, _unique([cite(r["source"]) for r in used]), used


def compare_answer(funds: list[str]) -> tuple[str, list[str], list[dict]] | None:
    """Every explicitly named fund side by side, each by its own reviewed ceiling."""
    compared = [(fund, *fund_summary(fund)) for fund in funds]
    compared = [(fund, text, rows) for fund, text, rows in compared if rows]
    if len(compared) < 2:
        return None
    used = [row for _, _, rows in compared for row in rows]
    answer = ("เทียบตามเพดานของหลักเกณฑ์กลาง: " + " | ".join(text for _, text, _ in compared)
              + " " + CEILING_IS_NOT_ALLOCATION)
    return answer, _unique([cite(r["source"]) for r in used]), used


def ranking_answer(direction: str) -> tuple[str, list[str], list[dict]] | None:
    """Which fund pays the most, or the least, per grant. Funds whose ceiling is only a yearly total have
    no per-grant figure to rank, and the answer says they were left out rather than guessing one."""
    rows = [f for f in store.FACTS
            if f["amount_type"] == "ceiling" and store.is_verified(f) and f["amount"]]
    if len(rows) < 3:
        return None
    ranked = sorted(rows, key=lambda f: f["amount"], reverse=(direction == "high"))[:3]
    name = lambda r: r["fund"] + (f" {r['tier']}" if r["tier"] else "")
    word = "สูงที่สุด" if direction == "high" else "ต่ำที่สุด"
    answer = (f"ตามเพดานของหลักเกณฑ์กลาง ทุนที่ให้ต่อทุน{word}คือ {name(ranked[0])} ไม่เกิน "
              f"{baht(ranked[0]['amount'])} บาทต่อทุน รองลงมาคือ {name(ranked[1])} "
              f"{baht(ranked[1]['amount'])} บาท และ {name(ranked[2])} {baht(ranked[2]['amount'])} บาท "
              "ทุนที่หลักเกณฑ์กลางกำหนดเฉพาะวงเงินรวมไม่ได้นับในอันดับนี้ " + CEILING_IS_NOT_ALLOCATION)
    return answer, _unique([cite(r["source"]) for r in ranked]), ranked


def details_answer(fund: str, clause_eight: dict | None, rule_pending: bool,
                   has_rule: bool) -> tuple[str, list[str], list[dict]] | None:
    """"มีเงื่อนไขอะไร": the rates and notes the criteria sets, then where the eligibility conditions live.

    Criteria clause 8 says conditions follow the college's announcements, so that is what the answer says
    when no reviewed rule is at hand, instead of the flat "ไม่มีกฎ" a reader cannot act on.
    """
    summary, used = fund_summary(fund)
    parts, citations = [], []
    if summary:
        parts.append(f"ตามหลักเกณฑ์กลาง {summary}")
        citations += [cite(r["source"]) for r in used]
    if has_rule:
        parts.append(f"ส่วนสิทธิ์ของคุณ ถามว่า ขอ{fund}ได้ไหม แล้วระบบจะตรวจตามกฎที่มีให้")
    elif rule_pending:
        parts.append("เงื่อนไขคุณสมบัติของทุนนี้อยู่ในประกาศประจำปีที่ยังรอเจ้าหน้าที่ตรวจและอนุมัติ จึงยังสรุปให้ไม่ได้")
    elif clause_eight:
        parts.append("เงื่อนไขคุณสมบัติและหลักเกณฑ์การพิจารณา หลักเกณฑ์กลางข้อ 8 ให้เป็นไปตามประกาศของวิทยาลัย "
                     "ซึ่งระบบยังไม่มีฉบับที่ตรวจแล้ว กรุณาสอบถามเจ้าหน้าที่กองทุน")
        citations.append(cite(clause_eight))
    if not parts:
        return None
    return " ".join(parts), _unique(citations), used


def applicant_status(profile: dict | None) -> tuple[bool, str] | None:
    """Whether the asker's record falls inside the applicant definition, without repeating the record.

    The answer text goes into the audit log, and employment type and affiliation are personal data under
    PDPA, so this says in or out and names the requirement, never the person's own values.
    """
    staff_type, department = (profile or {}).get("staff_type"), (profile or {}).get("department")
    if not staff_type or not department:
        return None
    if department != COLLEGE:
        return False, f"ตามข้อมูลที่ระบบมี คุณยังไม่อยู่ในกลุ่มผู้ขอรับทุน เพราะผู้ขอรับทุนต้องสังกัด{COLLEGE}"
    if staff_type in STAFF_TYPES:
        return True, "ตามข้อมูลที่ระบบมี คุณอยู่ในกลุ่มผู้ขอรับทุน"
    return False, "ตามข้อมูลที่ระบบมี ประเภทบุคลากรของคุณไม่อยู่ในนิยามผู้ขอรับทุน"


def who_may_apply(question: str, profile: dict | None) -> tuple[str, list[str]] | None:
    """Who may apply, quoted from the definitions and answered for the asker when they ask about
    themselves. A role the definition leaves out gets the condition, not a verdict about that person."""
    applicant, staff = store.define("ผู้ขอรับทุน"), store.define("บุคลากร")
    if not (applicant and staff):
        return None
    rule = f"ผู้ขอรับทุนคือบุคลากร ได้แก่{staff['text']}"
    citations = _unique([cite(c["chunk"]) for c in (applicant, staff)])
    personal_next = ("เงื่อนไขของทุนแต่ละตัวยังมีเพิ่มอีก ถ้าบอกชื่อทุน ระบบจะตรวจเฉพาะกฎที่ผ่านการยืนยันแล้ว "
                     "และจะแจ้งเมื่อข้อมูลยังไม่พอ")
    general_next = "เงื่อนไขของทุนแต่ละตัวยังมีเพิ่มอีก หากต้องการตรวจสิทธิ์ ให้ระบุชื่อทุนและข้อมูลของผู้ขอ"
    if understand.about_the_asker(question):
        status = applicant_status(profile)
        if status is None:
            return (f"ระบบยังไม่มีข้อมูลประเภทบุคลากรและสังกัดของคุณ จึงยังบอกไม่ได้ {rule} "
                    f"ถ้าคุณอยู่ในกลุ่มนี้ก็อาจเข้าข่ายผู้ขอรับทุน แต่ยังต้องตรวจเงื่อนไขรายทุน {personal_next}"), citations
        inside, sentence = status
        if inside:
            return f"{sentence} {personal_next}", citations
        return f"{sentence} {rule} ถ้าข้อมูลนี้ไม่ตรงกับความจริง กรุณาแจ้งเจ้าหน้าที่", citations
    person = understand.person_asked_about(question)
    if person in understand.OUTSIDE_ROLES:
        return f"{rule} ถ้า{person}ไม่ได้มีสถานะเป็นบุคลากรตามนี้ ก็ขอทุนไม่ได้", citations
    if person and person != "บุคลากร" and not any(person in t or t in person for t in STAFF_TYPES):
        return f"{rule} {person}ขอได้ถ้ามีสถานะเป็นบุคลากรตามนี้ {general_next}", citations
    return f"{rule} {general_next}", citations


def _usable_chunks() -> list[dict]:
    return [c for c in store.CHUNKS
            if c.get("status", "active") == "active" and store.is_verified(c)]


def funds_for_purpose(purpose: str, chunks: list[dict]) -> tuple[str, list[str], list[dict]]:
    """A purpose without a fund name cannot establish a grant or personal eligibility."""
    return ("ยังไม่แน่ใจว่ามีทุนใดรองรับวัตถุประสงค์นี้จากข้อมูลที่ตรวจแล้ว "
            "ยังไม่เปิดระบบแนะนำทุนจากลักษณะงานหรือพิจารณาสิทธิ์จากคำถามนี้ "
            "กรุณาถามว่า มีทุนอะไรบ้าง เพื่อดูรายการทั้งหมด หรือระบุชื่อทุนเพื่อถามข้อมูลตามเอกสาร"), [], []


def funds_about(topic: str, chunks: list[dict]) -> tuple[str, list[str], list[dict]]:
    """"ทุนเกี่ยวกับการสอนมีอะไรบ้าง": every fund whose reviewed clause holds the subject as written.

    The subject is matched whole in reviewed clauses. A subject most clauses mention, such as วิจัย,
    gets a few examples and a pointer to the full list. Wording no clause holds returns no match.
    """
    wanted = "".join(topic.split())
    matched = [c for c in _usable_chunks() if store.FUND_CLAUSE.match(c.get("clause", ""))
               and wanted in "".join(c["text"].split())]
    found = _unique([fund for clause in matched for fund in store.funds_mentioned_in([clause])])
    if not found:
        return NOT_FOUND.format(subject=topic), [], []
    displayed = found[:4] if len(found) > TOPIC_LIST_MAX else found
    citations = _unique([
        cite(clause)
        for fund in displayed
        if (clause := next((c for c in matched if fund in store.funds_mentioned_in([c])), None))
    ])
    if len(found) > TOPIC_LIST_MAX:
        return (f"ทุนที่เกี่ยวกับ{topic}มีหลายทุน เช่น {', '.join(found[:4])} ถามว่า มีทุนอะไรบ้าง "
                "เพื่อดูรายการทั้งหมด หรือระบุชื่อทุนที่สนใจ"), citations, []
    return f"ทุนที่เกี่ยวกับ{topic} ได้แก่ {', '.join(found)} {ASK_BY_NAME}", citations, []


def small_talk_answer(kind: str | None) -> tuple[str, list[str], list[dict]]:
    return SMALL_TALK.get(kind or "", SMALL_TALK["capability"]), [], []


def hypothetical_answer(received: str, target: str, ruling: dict,
                        rule_pending: bool = False,
                        premise_received: bool = True) -> tuple[str, list[str], list[dict]]:
    """"ถ้าได้ A แล้ว ขอ B ได้ไหม", from B's rules checked as if A were granted (pipeline._what_if_ruling).
    A refusal caused by the asker's own record is told apart from one the grant in question causes."""
    citations = [cite(ruling["source"])] if ruling.get("source") else []
    verdict, reason = ruling.get("verdict"), ruling.get("reason") or ""
    if (premise_received and verdict == "not_eligible"
            and str(ruling.get("field", "")).startswith("grants_this_year")):
        reason = reason.replace(f"ได้รับ{received}ไปแล้ว ", "", 1)
        return f"ไม่ได้ ถ้าได้รับ{received}แล้ว จะขอ{target}ไม่ได้ เพราะ{reason}", citations, []
    if verdict == "not_eligible":
        return f"ตามข้อมูลที่ระบบมี คุณยังขอ{target}ไม่ได้อยู่แล้ว เพราะ{reason}", citations, []
    if verdict == "eligible":
        if not premise_received:
            return (f"ได้ ตามกฎที่ระบบมี การที่ยังไม่เคยได้รับ{received}ไม่ทำให้ผิดเงื่อนไขของ{target} "
                    "ผลสุดท้ายต้องให้เจ้าหน้าที่กองทุนยืนยันอีกครั้ง"), citations, []
        return (f"ได้ ตามกฎที่ระบบมี การได้รับ{received}ไม่ได้ตัดสิทธิ์การขอ{target} "
                "ผลสุดท้ายต้องให้เจ้าหน้าที่กองทุนยืนยันอีกครั้ง"), citations, []
    if verdict == "unknown":
        return "ระบบยังไม่มีข้อมูลประเภทบุคลากรและสังกัดของคุณ จึงยังตรวจกรณีนี้ไม่ได้", citations, []
    if rule_pending:
        premise = f"การที่ยังไม่เคยได้รับ{received}" if not premise_received else f"การได้รับ{received}"
        return (f"เงื่อนไขของ{target}อยู่ในประกาศประจำปีที่ยังรอเจ้าหน้าที่ตรวจและอนุมัติ "
                f"จึงยังบอกไม่ได้ว่า{premise}มีผลต่อการขอ{target}หรือไม่"), [], []
    premise = f"การที่ยังไม่เคยได้รับ{received}" if not premise_received else f"การได้รับ{received}"
    return (f"ระบบยังไม่มีกฎที่ผ่านการตรวจซึ่งบอกได้ว่า{premise}มีผลต่อการขอ{target}หรือไม่ "
            "กรุณาสอบถามเจ้าหน้าที่กองทุน"), [], []


def name_notes(slots: dict) -> tuple[str, str]:
    """What an answer adds about the name asked, as (before, after). Before it: the fund has no tiers,
    when a tier was asked of a fund that has none. After it: which fund was answered, when the word used
    also names another. An answer that silently picked one of two funds, or dropped the tier asked,
    read as complete."""
    fund = slots.get("fund")
    if not fund or slots.get("intent") in NAME_NOTE_SKIP:
        return "", ""
    before = ""
    if slots.get("tier") and not any(f["tier"] for f in store.FACTS if f["fund"] == fund):
        before = f"ตามหลักเกณฑ์กลาง {fund}ไม่ได้แบ่งเป็นชั้น"
    others = slots.get("alternatives") or []
    after = f"คำตอบนี้เป็นของ{fund} ถ้าหมายถึง{' หรือ '.join(others)} ให้ถามโดยพิมพ์ชื่อเต็ม" if others else ""
    return before, after
