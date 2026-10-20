"""Compose grounded answers from reviewed facts, rules and document chunks."""

from app.modules.knowledge import corpus as store
from app.modules.knowledge.rules import describe_rules
from . import answers, understand
from .patterns import BROAD_ELIGIBILITY_RE
from app.modules.knowledge.formatting import baht as _baht, cite as _cite


def _defined_term(question: str) -> dict | None:
    """Longest defined term the question actually names. Short words such as ทุน match everything,
    so they are left out; a question that only says ทุน is asking which grant, not for a definition."""
    hits = [d for term, d in store.view().DEFINITIONS.items() if len(term) >= 4 and term in question]
    return max(hits, key=lambda d: len(d["term"])) if hits else None


def _catalogue_answer() -> tuple[str, list[str]]:
    """List the funds by their two headings. No amounts here: the list is for choosing, and a wall of
    numbers would bury that. The reader picks one and asks again."""
    catalogue = store.fund_catalogue()
    lines = []
    for group in ("ทุนส่งเสริมการวิจัย", "ทุนอุดหนุนกิจกรรม"):
        names = [f"{c['fund']} (ข้อ {c['clause']})" for c in catalogue if c["group"] == group]
        if names:
            lines.append(f"{group} มี {len(names)} รายการ คือ " + ", ".join(names))
    lines.append("ถามชื่อทุนที่สนใจต่อได้เลย ระบบจะตอบเพดานตามหลักเกณฑ์กลางหรือวงเงินจัดสรรรายปีที่เจ้าหน้าที่อนุมัติแล้ว หากยังรอตรวจจะแจ้งให้ทราบ")
    pages = sorted({row["source"]["page"] for row in catalogue if row.get("source", {}).get("page")})
    citations = [f"หลักเกณฑ์กลาง บัญชีแนบท้าย หน้า {page}" for page in pages]
    return " ".join(lines), citations


def _catalogue_group_summary() -> tuple[str, list[str]]:
    """A short orientation; the separate catalogue intent retains the complete list."""
    catalogue = store.fund_catalogue()
    groups = [group for group in ("ทุนส่งเสริมการวิจัย", "ทุนอุดหนุนกิจกรรม")
              if any(row["group"] == group for row in catalogue)]
    pages = sorted({row["source"]["page"] for row in catalogue if row.get("source", {}).get("page")})
    return " และ ".join(groups), [f"หลักเกณฑ์กลาง บัญชีแนบท้าย หน้า {page}" for page in pages]


def _cites(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(_cite(r["source"]) for r in rows))


def _clause_eight() -> dict | None:
    """Criteria clause 8: eligibility conditions follow the college's announcements."""
    return next((c for c in store.view().CHUNKS if c.get("doc") == "หลักเกณฑ์กลาง" and c.get("clause") == "ข้อ 8"
                 and c.get("status", "active") == "active" and store.is_verified(c)), None)


def _built_answer(slots: dict, ruling: dict) -> tuple[str, list[str], list[dict]] | None:
    """Questions about several funds, a ranking, tiers, conditions or a what-if (built in answers.py)."""
    intent, fund = slots.get("intent"), slots.get("fund")
    if intent == "compare":
        return answers.compare_answer(slots.get("funds", []))
    if intent == "rank":
        return answers.ranking_answer(slots["ranking"])
    if intent == "tiers" and fund:
        return answers.tiers_answer([fund, *slots.get("alternatives", [])])
    if intent == "details" and fund:
        usable = store.rules_for(fund, slots["fiscal_year"])
        if slots.get("criteria_request") and usable:
            described = describe_rules(usable)
            if described:
                text, citations = described
                return text, citations, []
        return answers.details_answer(fund, _clause_eight(), not usable and store.rules_waiting(fund), bool(usable))
    if intent == "hypothetical" and len(slots.get("funds", [])) >= 2:
        if slots.get("mixed_premise"):
            return ("คำถามนี้ระบุทั้งทุนที่เคยได้รับและไม่เคยได้รับหลายรายการในประโยคเดียว "
                    "ระบบจึงไม่เดาว่าเงื่อนไขใดเป็นของทุนใด กรุณาถามแยกทีละกรณี"), [], []
        *received_funds, target = slots["funds"]
        received = " และ ".join(received_funds)
        waiting = not store.rules_for(target, slots["fiscal_year"]) and store.rules_waiting(target)
        return answers.hypothetical_answer(received, target, ruling, waiting,
                                           slots.get("premise_received", True))
    return None


def _pending_notice(slots: dict) -> tuple[str, list[str], list[dict]]:
    """The annual figure waits for review, so give the reviewed central ceiling and say which is which.
    The ceiling's grant count, yearly total and rate note come too: without them "ให้กี่ทุน" and
    "วงเงินรวมเท่าไร" got a per-grant figure that answered neither."""
    waiting = (f"ส่วนวงเงินจัดสรรจริงของปีงบประมาณ {slots['fiscal_year']} อยู่ในประกาศที่ยังรอเจ้าหน้าที่ตรวจ"
               "และอนุมัติ จึงยังแสดงไม่ได้ ตัวเลขเพดานนี้ไม่ใช่วงเงินจัดสรรรายปี")
    if not slots.get("tier"):
        summary, rows = answers.fund_summary(slots["fund"])
        if any(r["tier"] for r in rows):
            return f"ตามหลักเกณฑ์กลาง {summary} {waiting}", _cites(rows), rows
    rows = [f for f in store.fact_lookup(slots["fund"], slots["tier"], None, "ceiling")
            if store.is_verified(f) and f["amount"]]
    if rows:
        row, tier = rows[0], f" {rows[0]['tier']}" if rows[0]["tier"] else ""
        text = f"{row['fund']}{tier}มีเพดานตามหลักเกณฑ์กลาง " + " ".join(answers.ceiling_terms(row))
        return f"{text} {waiting}", [_cite(row["source"])], [row]
    return ("พบประกาศจัดสรรของทุนนี้แล้ว แต่ข้อมูลจากเอกสารสแกนยังรอเจ้าหน้าที่ตรวจและอนุมัติ "
            "ระบบจึงยังไม่แสดงวงเงินจากประกาศฉบับนี้"), [], []


def compose_answer(slots: dict, evidence: dict, ruling: dict,
                   profile: dict | None = None) -> tuple[str, list[str], list[dict]]:
    facts = evidence["facts"]
    if slots["intent"] == "clarify_reference":
        conflict = slots["clause_conflict"]
        by_clause, by_name = conflict["clause_row"], conflict["name_row"]
        return (f"ตามบัญชีแนบท้าย ข้อ {by_clause['clause']} คือ{by_clause['fund']} "
                f"ส่วน{by_name['fund']}อยู่ข้อ {by_name['clause']} "
                "ชื่อทุนกับเลขข้อที่ระบุจึงไม่ตรงกัน คุณต้องการถามทุนใด?",
                _cites([by_clause, by_name]), [])
    built = _built_answer(slots, ruling)
    if built:
        return built
    if slots["intent"] == "smalltalk":
        return answers.small_talk_answer(slots.get("chatter"))
    if slots["intent"] == "topic":
        return answers.funds_about(slots["topic"], evidence["chunks"])
    if slots["intent"] == "history":
        return ("ระบบเดโม่ยังไม่ได้เชื่อมทะเบียนประวัติการยื่นขอและการได้รับทุนของผู้ใช้ "
                "จึงยังบอกไม่ได้ว่าคุณเคยขอหรือได้รับทุนใดบ้าง "
                "ข้อมูลสถานะที่มีอยู่ใช้ตรวจเงื่อนไขบางข้อเท่านั้น ไม่ใช่ประวัติทุนที่ครบถ้วน"), [], []
    if slots["intent"] == "open_now":
        return ("ยังบอกไม่ได้ว่าทุนใดเปิดรับอยู่ตอนนี้ เพราะระบบไม่มีข้อมูลสถานะเปิดรับที่ผ่านการตรวจยืนยัน "
                "รายการทุนในหลักเกณฑ์กลางไม่ได้หมายความว่าทุกทุนกำลังเปิดรับ "
                "ถ้าต้องการดูรายการทุนตามหลักเกณฑ์กลาง ให้ถามว่า มีทุนอะไรบ้าง"), [], []
    if slots["intent"] == "recommendation":
        lead = "จากคำถามก่อนหน้าที่คุณถามถึงรายการทุน " if slots["catalogue_context"] else ""
        return (lead + "ตอนนี้ยังไม่เปิดระบบแนะนำทุนหรือเลือกทุนให้เฉพาะบุคคล "
                "ระบบบอกข้อมูลจากเอกสารที่ตรวจแล้วได้ กรุณาถามว่า มีทุนอะไรบ้าง เพื่อดูรายการทั้งหมด "
                "หรือระบุชื่อทุนที่สนใจเพื่อถามรายละเอียด เพดาน และเงื่อนไข"), [], []
    if slots["intent"] == "catalogue_details":
        return answers.catalogue_details_answer()
    if slots["intent"] == "list":
        text, citations = _catalogue_answer()
        return text, citations, []          # code built this from verified rows; no model touched it
    if slots.get("prerequisite"):
        return (f"ระบบยังไม่มีข้อมูลที่ผ่านการตรวจยืนยันว่าต้องเคยได้รับทุนอื่นก่อนจึงจะขอ"
                f"{slots['fund']}ได้ จึงยังสรุปเงื่อนไขข้อนี้ไม่ได้ กรุณาตรวจประกาศของทุนหรือสอบถามเจ้าหน้าที่"), [], []
    if slots["fund"] and slots["intent"] == "definition":
        catalogue = next((row for row in store.fund_catalogue() if row["fund"] == slots["fund"]), None)
        ceiling = next((row for row in store.fact_lookup(slots["fund"], None, None, "ceiling")
                        if store.is_verified(row)), None)
        if catalogue and ceiling:
            return (f"ตามบัญชีแนบท้ายของหลักเกณฑ์กลาง {slots['fund']}อยู่ในหมวด"
                    f"{catalogue['group']} ถ้าต้องการทราบเพดานหรือเงื่อนไขของทุนนี้ "
                    "ถามต่อโดยระบุเรื่องที่สนใจได้"), [_cite(ceiling["source"])], []
    if slots["fund"] and slots["intent"] == "match":
        ceiling = next((row for row in store.fact_lookup(slots["fund"], None, None, "ceiling")
                        if store.is_verified(row)), None)
        citations = [_cite(ceiling["source"])] if ceiling else []
        return (f"คำถามนี้ตรงกับ{slots['fund']}ในหลักเกณฑ์กลาง "
                "หากต้องการทราบเพดานหรือเงื่อนไข ให้ถามต่อโดยระบุเรื่องที่ต้องการ"), citations, []
    if slots["fund"] is None and slots["intent"] == "eligibility":
        question = slots["standalone_question"]
        purpose = understand.purpose_of(question)
        if purpose and not BROAD_ELIGIBILITY_RE.search(question):
            return answers.funds_for_purpose(purpose, evidence["chunks"])
        answered = answers.who_may_apply(question, profile)
        if answered:
            text, citations = answered
            if BROAD_ELIGIBILITY_RE.search(question) and understand.about_the_asker(question):
                catalogue, catalogue_citations = _catalogue_group_summary()
                text = ("ยังสรุปไม่ได้ว่าคุณขอทุนใดได้บ้าง เพราะต้องตรวจเงื่อนไขของแต่ละทุนแยกกัน\n\n"
                        "หลักเกณฑ์กลางแบ่งรายการเป็น " + catalogue + "\n\n" + text
                        + " หากต้องการข้อมูลเฉพาะทุน กรุณาระบุชื่อทุน หรือถามว่า มีทุนอะไรบ้าง "
                        "เพื่อดูรายการทั้งหมด")
                citations = list(dict.fromkeys(catalogue_citations + citations))
            return text, citations, []
    if slots["fund"] is None and slots["intent"] == "definition":
        term = _defined_term(slots["standalone_question"])
        if term:
            return (f"ตามนิยามในประกาศ {term['term']} หมายความว่า {term['text']}"), [_cite(term["chunk"])], []
    if slots["fund"] is None:
        if slots.get("unresolved_reference"):
            return ("คำถามนี้อ้างถึงทุนก่อนหน้า แต่ในบทสนทนานี้ยังไม่มีชื่อทุนตัวเดียวที่ยืนยันได้ "
                    "ระบบจึงยังไม่รู้ว่าหมายถึงทุนใด ลองถามว่า มีทุนอะไรบ้าง เพื่อดูรายการทั้งหมด "
                    "หรือพิมพ์ชื่อทุนที่ต้องการทราบมาตรง ๆ"), [], []
        return ("ระบบยังไม่แน่ใจว่าคำถามนี้หมายถึงทุนใด ถามว่า มีทุนอะไรบ้าง เพื่อดูรายการทั้งหมด "
                "หรือระบุชื่อทุนพร้อมปีงบประมาณที่ต้องการทราบ"), [], []
    if slots["intent"] == "eligibility" and not slots.get("about_asker", True):
        answered = answers.who_may_apply(slots["standalone_question"], None)
        if answered:
            text, citations = answered
            return ("ระบบจะไม่ใช้โปรไฟล์ที่เลือกอยู่ตัดสินสิทธิ์แทนบุคคลอื่น " + text), citations, []
    if slots["intent"] == "application":
        return (f"ระบบยังไม่มีข้อมูลที่ผ่านการตรวจยืนยันเกี่ยวกับขั้นตอนสมัคร เอกสาร กำหนดรับสมัคร "
                f"การประกาศผล การเบิกจ่าย การคืนเงิน หรือการถอนคำขอของ{slots['fund']} "
                "จึงยังตอบเรื่องนี้ไม่ได้ กรุณาตรวจประกาศรับสมัครหรือสอบถามเจ้าหน้าที่"), [], []
    if slots["intent"] == "unknown":
        return (f"ระบบเข้าใจว่าคุณถามถึง{slots['fund']} แต่ยังไม่แน่ใจว่าต้องการทราบเรื่องใด "
                "ลองระบุว่าอยากทราบเพดาน วงเงินปีงบประมาณ เงื่อนไข หรือวิธีสมัคร"), [], []
    if slots["intent"] == "eligibility" and ruling["verdict"] == "no_rule":
        status = answers.applicant_status(profile)
        known = f" ข้อที่ยืนยันได้ตอนนี้คือ{status[1].replace('ตามข้อมูลที่ระบบมี', '', 1)}" if status else ""
        definition = answers.who_may_apply("ฉันขอทุนได้ไหม", profile)
        definition_citations = definition[1] if status and definition else []
        if evidence.get("pending_review"):
            return (f"เงื่อนไขของ{slots['fund']}อยู่ในประกาศประจำปีที่ยังรอเจ้าหน้าที่ตรวจและอนุมัติ "
                    f"จึงยังสรุปสิทธิ์ไม่ได้{known}"), definition_citations, []
        return (f"ระบบยังไม่มีกฎคุณสมบัติของ{slots['fund']}ที่ผ่านการตรวจยืนยัน จึงยังสรุปสิทธิ์ไม่ได้{known} "
                "กรุณาตรวจประกาศของทุนหรือสอบถามเจ้าหน้าที่"), definition_citations, []
    if not facts:
        if evidence.get("pending_review"):
            return _pending_notice(slots)
        return ("ข้อมูลไม่เพียงพอ ระบบไม่พบหลักฐานในเอกสารที่ตรงกับคำถามนี้ "
                "กรุณาระบุชื่อทุนและปีงบประมาณให้ชัดเจนขึ้น"), [], []
    fact = facts[0]
    used = [fact]
    parts = []
    tier = f" {fact['tier']}" if fact["tier"] else ""
    year = slots["fiscal_year"]
    if fact["amount_type"] == "ceiling":
        summary, tier_rows = answers.fund_summary(fact["fund"]) if not slots.get("tier") else ("", [])
        if any(r["tier"] for r in tier_rows):
            # A tiered fund asked about without a tier. Its whole-fund row carries only the yearly total,
            # and read alone it once made the answer say the fund had no tiers.
            used = tier_rows
            if evidence.get("ceiling_fallback"):
                parts.append(f"ระบบยังไม่มีข้อมูลวงเงินจัดสรรของ{fact['fund']} ปีงบประมาณ {year} "
                             "จึงตอบตามเพดานของหลักเกณฑ์กลาง")
            parts += [f"ตามหลักเกณฑ์กลาง {summary}", answers.CEILING_IS_NOT_ALLOCATION]
        elif evidence.get("ceiling_fallback"):
            # The system knows only that it holds no annual row. Whether an announcement exists is a fact
            # it cannot see, and "ยังไม่มีประกาศจัดสรร" once denied one that listed the very tier asked.
            if fact["amount"]:
                parts.append(f"{fact['fund']}{tier}ขอได้ไม่เกิน {_baht(fact['amount'])} บาทต่อทุน "
                             "ตามเพดานของหลักเกณฑ์กลาง")
            else:
                parts.append(f"หลักเกณฑ์กลางไม่ได้กำหนดวงเงินต่อทุนของ{fact['fund']} กำหนดเฉพาะวงเงินรวม")
            parts += answers.ceiling_terms(fact, per_grant=False)
            parts.append(f"ระบบยังไม่มีข้อมูลวงเงินจัดสรรของปีงบประมาณ {year} "
                         "ตัวเลขนี้จึงเป็นเพดาน ไม่ใช่วงเงินที่จัดสรรจริง")
        else:
            if fact["amount"]:
                parts.append(f"เพดานตามหลักเกณฑ์กลางของ{fact['fund']}{tier} คือ {_baht(fact['amount'])} บาทต่อทุน")
            else:
                parts.append(f"หลักเกณฑ์กลางกำหนดเฉพาะวงเงินรวมของ{fact['fund']} ไม่ได้กำหนดวงเงินต่อทุน")
            parts += answers.ceiling_terms(fact, per_grant=False)
            parts.append("ตัวเลขนี้เป็นเพดานสูงสุด ไม่ใช่วงเงินที่จัดสรรจริงของปีงบประมาณใดปีหนึ่ง")
    else:
        # the announcement heads this column "วงเงินสูงสุดต่อทุน", so the figure is a cap on what may
        # be granted, never a promise of what will be. Saying "ได้ X" reads as the latter and the
        # number gate cannot catch it, because the digits are right and only the meaning moved.
        parts.append(f"{fact['fund']}{tier} ปีงบประมาณ {fact['fiscal_year']} "
                     f"ขอได้สูงสุด {_baht(fact['amount'])} บาทต่อทุน")
        if fact["max_grants_per_person"]:
            parts.append(f"ขอได้คนละไม่เกิน {fact['max_grants_per_person']} ทุนต่อปี")
        if fact.get("max_grants_per_year"):
            parts.append(f"ประกาศปีงบประมาณนี้ให้ได้ไม่เกิน {fact['max_grants_per_year']} ทุน")
        if fact["total_cap"]:
            parts.append(f"วงเงินรวมไม่เกิน {_baht(fact['total_cap'])} บาท")
        if fact.get("note"):                     # the condition that makes the figure true, next to it
            parts.append(fact["note"])
        ceiling = [row for row in store.fact_lookup(fact["fund"], fact["tier"], None, "ceiling")
                   if store.is_verified(row)]
        if ceiling and ceiling[0]["amount"] and ceiling[0]["amount"] != fact["amount"]:
            used.append(ceiling[0])
            parts.append(f"ตัวเลขนี้เป็นวงเงินที่จัดสรรจริงของปีงบ {fact['fiscal_year']} "
                         f"ไม่ใช่เพดาน {_baht(ceiling[0]['amount'])} บาทตามหลักเกณฑ์กลาง")
    if ruling["verdict"] == "eligible":
        parts.append(f"ตามข้อมูลผู้ใช้ที่ระบบมี คุณเข้าเกณฑ์ขอทุนนี้ในปีงบ {slots['fiscal_year']}")
    elif ruling["verdict"] == "not_eligible":
        parts.append(f"ตามข้อมูลผู้ใช้ที่ระบบมี คุณยังไม่เข้าเกณฑ์ เพราะ{ruling['reason']}")
    elif ruling["verdict"] == "unknown":
        parts.append("ระบบยังไม่มีข้อมูลประเภทบุคลากรและสังกัดของคุณ จึงยังสรุปสิทธิ์ไม่ได้ "
                     "ถ้าคุณเป็นบุคลากรสังกัดวิทยาลัยการคอมพิวเตอร์และยังไม่ได้รับทุนนี้ในปีนี้ ก็เข้าเกณฑ์ตามประกาศ")
    if ruling["verdict"] in ("eligible", "not_eligible", "unknown"):
        parts.append("ผลสุดท้ายต้องให้เจ้าหน้าที่กองทุนยืนยันอีกครั้ง")
    if slots["year_assumed"] and fact["amount_type"] == "allocated":
        parts.append(f"คำถามไม่ได้ระบุปี ระบบจึงตอบของปีงบประมาณ {slots['fiscal_year']} เป็นค่าเริ่มต้น")
    if evidence.get("ceiling_fallback"):       # last, so the rate note stays next to the figures it qualifies
        parts.append("ถ้าต้องการวงเงินที่จัดสรรจริงของปีนี้ ให้ตรวจประกาศประจำปีหรือสอบถามเจ้าหน้าที่กองทุน")
    citations = [_cite(f["source"]) for f in used]
    if ruling["source"] and _cite(ruling["source"]) not in citations:
        citations.append(_cite(ruling["source"]))
    return " ".join(parts), citations, used
