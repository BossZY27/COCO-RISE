"""Question analysis shared by the single API pipeline and offline tools."""
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo
from app.modules.knowledge import corpus as store
from . import understand
from .patterns import BROAD_ELIGIBILITY_RE, GENERIC_CATALOGUE_RE, PERSONAL_HISTORY_RE, REPEAT_ELIGIBILITY_RE

# The fiscal year a question means when it names none. Thai fiscal years turn over on 1 October, and the
# data on hand is for 2569, so the operator moves this with RISE_FISCAL_YEAR when a new announcement lands.
DEFAULT_YEAR = int(os.getenv("RISE_FISCAL_YEAR", "2569"))


def thai_fiscal_year(now: datetime | None = None) -> int:
    """Thai fiscal year in Bangkok: it changes on 1 October, not 1 January."""
    local = now or datetime.now(ZoneInfo("Asia/Bangkok"))
    return local.year + (544 if local.month >= 10 else 543)


# Relative words such as ปีนี้ follow the real calendar.  A question that says no
# year still uses DEFAULT_YEAR, the latest reviewed annual dataset, and the answer
# explicitly labels that assumption.  Compute the real year per request so a
# long-running server does not stay on the old fiscal year after 1 October.
# A year stands alone: "25000" holds "2500", which was once read as fiscal year 2500.
YEAR_RE = re.compile(r"(?<!\d)(25\d\d)(?!\d)")
# yes-or-no endings such as ได้ไหม and ได้หรือเปล่า are read by understand.asks_if_possible
ELIGIBILITY_WORDS = ("มีสิทธิ", "สิทธิ์", "คุณสมบัติ", "เงื่อนไข", "ขอเพิ่ม", "เพิ่มอีก", "ใคร", "ขอได้บ้าง",
                     "ผู้ขอรับทุน")
CEILING_WORDS = ("เพดาน", "หลักเกณฑ์กลาง")
AMOUNT_WORDS = ("เท่าไร", "เท่าไหร่", "กี่บาท", "กี่ทุน", "วงเงิน", "จำนวนเงิน", "ทุนละ", "ได้เงิน",
                "ได้ค่าตอบแทน")
DEFINITION_WORDS = ("คืออะไร", "คืออะไรบ้าง", "คือทุนอะไร", "คือใคร", "หมายความว่า", "หมายถึง", "นิยาม", "แปลว่า")
FUND_CATEGORY_WORDS = ("อยู่หมวดไหน", "อยู่หมวดอะไร", "อยู่ในหมวดไหน", "อยู่ในหมวดอะไร",
                       "เป็นทุนประเภทไหน", "เป็นทุนประเภทอะไร", "จัดอยู่ในหมวด")
# Answers whose wording code assembles from several rows. A rephrasing that dropped a fund or a figure
# would still read as complete, so they never go to the model.
APPLICATION_WORDS = ("สมัครยังไง", "สมัครอย่างไร", "ยื่นยังไง", "ยื่นอย่างไร", "เอกสารอะไร",
                     "เอกสารที่ต้องใช้", "ขั้นตอน", "เปิดรับ", "ปิดรับ", "ส่งที่ไหน", "สมัครที่ไหน",
                     "ยื่นที่ไหน", "ยื่นถึง", "หมดเขต", "วันสุดท้าย", "ประกาศผล", "ผลออก",
                     "สถานะคำขอ", "อนุมัติหรือยัง", "เบิกยังไง", "เบิกอย่างไร", "เบิกจ่าย",
                     "โอนเงิน", "คืนเงิน", "คืนทุน", "ชดใช้", "ถอนทุน", "ถอนคำขอ", "ยกเลิกคำขอ")
# People commonly put the fund name between the action and the question word
# ("สมัครทุน...ยังไง"), so exact substrings alone miss ordinary phrasing.
APPLICATION_RE = re.compile(
    r"(?:(?:สมัคร|ยื่น|ส่ง).{0,100}?(?:ยังไง|อย่างไร|ได้ที่ไหน|ที่ไหน|ได้ถึง(?:วัน)?ไหน|ถึง(?:วัน)?ไหน|เมื่อไร|วันไหน|ทางไหน|ต้องมีอะไร|ต้องทำอะไร)"
    r"|(?:ได้เงิน|รับเงิน|โอนเงิน|อนุมัติ).{0,60}?(?:เมื่อไร|ตอนไหน|วันไหน)"
    r"|เอกสาร.{0,60}?(?:อะไร|ต้องมี|ใช้อะไร|ส่ง|ทางไหน)"
    r"|(?:ต้องใช้|ต้องมี).{0,40}?(?:เอกสาร|อะไร)"
    r"|ต้องทำอะไร.{0,80}?(?:สมัคร|ยื่น))"
)
PREREQUISITE_WORDS = ("ต้องเคย", "จำเป็นต้องเคย", "ก่อนถึงจะขอ", "ก่อนจึงจะขอ")
# "มีทุนอะไรบ้าง" is the first question anyone asks, and the catalogue answers it from rows we trust
LIST_WORDS = ("มีทุนอะไร", "มีทุนไรบ้าง", "ทุนไรบ้าง", "ทุนอะไรบ้าง", "มีทุนกี่", "ทุนมีกี่", "ทุนทั้งหมด", "รายการทุน",
              "ทุนมีอะไร", "มีกองทุนอะไร", "ทุนไหนบ้าง", "ทุนตัวไหนบ้าง", "ทุนตัวใดบ้าง",
              "ทุนมีตัวไหนบ้าง", "ทุนแต่ละตัวคืออะไร", "ทุนเเต่ละตัวคืออะไร")
RECOMMENDATION_WORDS = ("แนะนำ", "ช่วยเลือก", "ช่วยคัด", "เหมาะกับ", "เข้ากับ", "ควรขอ", "ควรสมัคร")
CURRENT_OPEN_WORDS = ("เปิดตอนนี้", "เปิดอยู่", "กำลังเปิด", "เปิดรับตอนนี้", "เปิดรับอยู่")
# "ทุนนี้" points back at the previous turn, so the current question names no fund of its own
ANAPHORA = ("ทุนนี้", "ทุนวิจัยนี้", "ทุนดังกล่าว", "ทุนนั้น", "ทุนตัวนี้",
           "มันคือทุน", "มันคืออะไร", "อันนี้คือทุน", "อันนี้คืออะไร")
CLAUSE_REFERENCE_RE = re.compile(r"ข้อ\s*(?:ที่\s*)?([123]\.\d{1,2})(?![\d.])")


def _wants_catalogue_details(question: str, catalogue_context: bool, mentioned: list[dict]) -> bool:
    """Distinguish 'tell me about each grant' from a plain list or one named grant."""
    if mentioned:
        return False
    broad = any(word in question for word in ("ทุนแต่ละตัว", "ทุนเเต่ละตัว", "ทุนทุกตัว", "ทุกทุน",
                                               "ทุนทั้งหมด", "ทุนตัวไหนทำ", "ทุนตัวใดทำ"))
    followup = catalogue_context and "แต่ละตัว" in question
    detail = any(word in question for word in ("ข้อมูล", "รายละเอียด", "อธิบาย", "สรุป", "แจกแจง",
                                                "ทำอะไร", "ทำไร", "ทำหน้าที่", "ดู", "ขอทั้งหมด", "ยังไง"))
    return detail and (broad or followup)


@contextmanager
def _stage(times: dict, name: str):
    t0 = time.perf_counter()
    yield
    times[name] = round((time.perf_counter() - t0) * 1000, 1)


# ---------------------------------------------------------------- step 1: analyse
def _match_fund(text: str) -> tuple[str | None, float]:
    exact = [f for f in store.view().FUNDS if f in text]
    if exact:
        return max(exact, key=len), 1.0
    best, best_ratio = None, 0.0
    for fund in store.view().FUNDS:
        n = len(fund)
        for width in (n - 4, n, n + 4):
            if width < 4:
                continue
            for i in range(max(1, len(text) - width + 1)):
                ratio = SequenceMatcher(None, fund, text[i:i + width]).ratio()
                if ratio > best_ratio:
                    best, best_ratio = fund, ratio
    return (best, best_ratio) if best_ratio >= 0.80 else (None, best_ratio)


def _find_fund(text: str) -> tuple[str | None, float, list[str]]:
    """A full name or a reviewed alias first, then the fuzzy match. Returns the fund, how sure, and the
    other funds an ambiguous word could also mean."""
    named = understand.mentions(text)
    if named:
        return named[0]["fund"], 1.0 if named[0]["exact"] else 0.95, named[0]["alternatives"]
    fund, confidence = _match_fund(text)
    return fund, confidence, []


def _fiscal_year(question: str, previous: str) -> tuple[int, bool]:
    """The year the question means: a written year, then ปีหน้า or ปีงบ 69, then the previous turn's."""
    for text in (question, previous):
        if not text:
            continue
        written = YEAR_RE.search(text)
        if written:
            return int(written.group(1)), False
        relative = understand.relative_year(text, thai_fiscal_year())
        if relative:
            return relative, False
    return DEFAULT_YEAR, True


def _find_tier(text: str) -> str | None:
    return next((store.TIER_ALIAS.get(t, t) for t in store.TIERS if t in text), None)


def _relation_funds(question: str, mentioned: list[dict]) -> list[str]:
    """Return every stated premise followed by the requested target."""
    funds = [m["fund"] for m in mentioned]
    if len(funds) < 2:
        return funds
    starts = [question.rfind(word) for word in ("ขอ", "สมัคร", "ยื่น")]
    request_at = max(starts)
    requested = understand.mentions(question[request_at:]) if request_at >= 0 else []
    target = requested[0]["fund"] if requested else funds[-1]
    premises = [fund for fund in funds if fund != target]
    return [*premises, target]


def _conflicting_clause(question: str, mentioned: list[dict]) -> dict | None:
    """Catch one explicit annex clause paired with a different, unambiguous fund name."""
    if "ประกาศ" in question and not any(word in question for word in ("หลักเกณฑ์กลาง", "บัญชีแนบท้าย")):
        return None
    clauses = CLAUSE_REFERENCE_RE.findall(question)
    if len(clauses) != 1 or len(mentioned) != 1 or mentioned[0]["alternatives"]:
        return None
    catalogue = store.fund_catalogue()
    by_clause = next((row for row in catalogue if row["clause"] == clauses[0]), None)
    by_name = next((row for row in catalogue if row["fund"] == mentioned[0]["fund"]), None)
    if not by_clause or not by_name or by_clause["fund"] == by_name["fund"]:
        return None
    return {"clause_row": by_clause, "name_row": by_name}


def _explicit_annex_fund(question: str, mentioned: list[dict],
                         catalogue_context: bool = False) -> str | None:
    """Resolve a clause only with an explicit or conversation-established central annex."""
    if mentioned or not (catalogue_context or any(
            word in question for word in ("หลักเกณฑ์กลาง", "บัญชีแนบท้าย"))):
        return None
    clauses = CLAUSE_REFERENCE_RE.findall(question)
    if len(clauses) != 1:
        return None
    matches = [row for row in store.fund_catalogue() if row["clause"] == clauses[0]]
    return matches[0]["fund"] if len(matches) == 1 else None


def analyze(question: str, history: list[str] | None = None, profile: dict | None = None,
            context: dict | None = None, use_llm: bool = False, slot_extractor=None, default_year: int | None = None) -> dict:
    """Fill the slots the rest of the pipeline needs.

    A follow-up such as "แล้วชั้นกลางล่ะ" only makes sense with the previous turn, so missing slots fall back
    to the last question. What the current question says always wins. An LLM is called only when the rules
    cannot find the fund at all.
    """
    question = understand.normalize(question)
    history = [understand.normalize(turn) for turn in history or []]
    previous = history[-1] if history else ""
    context = context if isinstance(context, dict) else {}
    context_fund = context.get("fund") if context.get("fund") in store.view().FUNDS else None
    # "ทุนนี้" says "the grant we were just talking about", so the name comes from the conversation
    # and never from a fuzzy guess. Remember that it was asked, because a question pointing back at
    # nothing needs a different answer from a question that simply named no fund.
    mentioned = understand.mentions(question)
    clause_conflict = _conflicting_clause(question, mentioned)
    catalogue_context = any(
        any(word in earlier for word in LIST_WORDS) or GENERIC_CATALOGUE_RE.search(earlier)
        for earlier in history
    )
    open_now = any(word in question for word in CURRENT_OPEN_WORDS)
    recommendation_request = not any(m["exact"] for m in mentioned) and (
        (("ทุน" in question or catalogue_context)
         and any(word in question for word in RECOMMENDATION_WORDS))
        or (question.startswith(("อยาก", "ต้องการ", "สนใจ"))
            and not any(word in question for word in LIST_WORDS)
            and not GENERIC_CATALOGUE_RE.search(question)
            and not any(word in question for word in
                        ("เพดาน", "วงเงิน", "เท่าไร", "เท่าไหร่", "เงื่อนไข", "รายละเอียด")))
    )
    catalogue_details_request = _wants_catalogue_details(question, catalogue_context, mentioned)
    # An explicit name in the same sentence resolves "ทุนนี้" immediately.  Only an
    # otherwise unnamed reference may fall back to an earlier conversation turn.
    points_back = not mentioned and any(a in question for a in ANAPHORA)
    repeat_eligibility = bool(REPEAT_ELIGIBILITY_RE.search(question))
    prerequisite = any(word in question for word in PREREQUISITE_WORDS)
    relation_question = (len(mentioned) >= 2
                         and any(word in question for word in ("ขอ", "สมัคร", "ยื่น"))
                         and any(word in question for word in ("เคย", "ได้รับ", "ได้มาแล้ว", "ได้เงิน"))
                         and ("ถ้า" in question or "หาก" in question or repeat_eligibility
                              or understand.asks_if_possible(question)))
    hypothetical = not prerequisite and relation_question
    premise_received = not any(word in question for word in ("ไม่เคย", "ยังไม่เคย", "ไม่เคยได้รับ"))
    mixed_premise = (
        len(mentioned) > 2
        and not premise_received
        and bool(re.search(r"(?:แต่|และ)\s*(?:เคย)?(?:ได้รับ|ได้เงิน|ได้มาแล้ว)", question))
    )
    application_question = (open_now or any(w in question for w in APPLICATION_WORDS)
                            or bool(APPLICATION_RE.search(question)))
    explicit_amount = any(w in question for w in AMOUNT_WORDS if w != "ได้เงิน")
    explicit_discourse_followup = (
        question.startswith(("แล้ว", "ส่วน"))
        and question.rstrip(" ?").endswith(("ล่ะ", "ละ", "บ้าง"))
    )
    constrained_followup = (
        points_back
        or explicit_discourse_followup
        or _find_tier(question) is not None
        or application_question
        or understand.wants_details(question)
        or understand.asks_if_possible(question)
        or any(word in question for word in (
            "ขอได้เท่าไร", "ได้เท่าไร", "กี่บาท", "กี่ทุน", "วงเงิน", "เพดาน", "ทุนละ",
            "ปีหน้า", "ปีนี้", "ปีที่แล้ว", "เงื่อนไข", "แต่ละชั้น", "แต่ละระดับ",
            "ใครขอได้", "ใครมีสิทธิ์", "ใครมีสิทธิ",
        ))
    )
    personal_history = (bool(PERSONAL_HISTORY_RE.search(question))
                        and not hypothetical
                        and not repeat_eligibility
                        and "ไม่เคย" not in question
                        and not prerequisite
                        and not explicit_amount
                        and not any(w in question for w in CEILING_WORDS)
                        and not application_question)
    broad_eligibility = not personal_history and bool(BROAD_ELIGIBILITY_RE.search(question))
    named = [] if points_back or broad_eligibility or personal_history or recommendation_request else mentioned
    relation_funds = _relation_funds(question, named) if hypothetical else [m["fund"] for m in named]
    if prerequisite and len(named) >= 2:
        # The requested fund, not the prerequisite mentioned first, is the subject.
        target = _relation_funds(question, named)[-1]
        named = [next(m for m in named if m["fund"] == target)]
    if points_back or broad_eligibility or personal_history or recommendation_request:
        fund, confidence, alternatives = None, 0.0, []
    else:
        fund, confidence, alternatives = _find_fund(question)
        annex_fund = _explicit_annex_fund(question, mentioned, bool(catalogue_context))
        if annex_fund:
            fund, confidence, alternatives = annex_fund, 1.0, []
        if prerequisite and named:
            fund = named[0]["fund"]
            confidence = 1.0 if named[0]["exact"] else 0.95
            alternatives = named[0]["alternatives"]
    from_current = fund is not None
    for earlier in reversed(history or []):      # a vague turn in between must not break the thread
        if broad_eligibility or personal_history:
            break
        if not constrained_followup:
            break
        if CLAUSE_REFERENCE_RE.search(question):
            # A new clause is a new subject; never borrow the previous clause's fund.
            break
        if fund is not None:
            break
        earlier_conflict = _conflicting_clause(earlier, understand.mentions(earlier))
        if earlier_conflict:
            # Until the user picks one side, neither name may become the next answer's subject.
            clause_conflict = earlier_conflict
            break
        fund, confidence, alternatives = _find_fund(earlier)
        annex_fund = _explicit_annex_fund(earlier, understand.mentions(earlier))
        if annex_fund:
            fund, confidence, alternatives = annex_fund, 1.0, []
    used_context = bool(context_fund and not from_current and constrained_followup
                        and not broad_eligibility and not personal_history and not recommendation_request
                        and not clause_conflict and not CLAUSE_REFERENCE_RE.search(question))
    if used_context:
        fund, confidence, alternatives = context_fund, 1.0, []
    chatter = None if from_current else understand.small_talk(question)
    topic = "" if from_current else understand.topic_of(question)
    if chatter or topic:        # "ขอบคุณครับ" after a grant question is not another question about it
        fund, confidence, alternatives = None, 0.0, []
        used_context = False
    context_tier = context.get("tier")
    if context_tier not in store.TIERS and context_tier not in store.TIER_ALIAS.values():
        context_tier = None
    tier = _find_tier(question) or (context_tier if used_context else
                                  _find_tier(previous) if previous and not from_current else None)
    fiscal_year, assumed = _fiscal_year(question, "" if used_context else previous)
    if assumed and default_year is not None:
        fiscal_year = default_year
    context_year = context.get("fiscal_year")
    if (used_context and assumed and context.get("year_assumed") is False
            and type(context_year) is int and 2500 <= context_year <= 2700):
        fiscal_year, assumed = context_year, False
    ranking = understand.wants_ranking(question) if fund is None else None
    if clause_conflict:
        intent = "clarify_reference"
    elif chatter:
        intent = "smalltalk"
    elif topic and not recommendation_request:
        intent = "topic"
    elif personal_history:
        intent = "history"
    elif open_now and not mentioned:
        intent = "open_now"
    elif broad_eligibility:
        intent = "eligibility"
    elif hypothetical:
        intent = "hypothetical"
    elif len(named) >= 2 and (understand.wants_comparison(question) or explicit_amount):
        intent = "compare"
    elif ranking:
        intent = "rank"
    elif recommendation_request:
        intent = "recommendation"
    elif catalogue_details_request and not application_question:
        intent = "catalogue_details"
    elif fund and understand.wants_tier_list(question):
        intent = "tiers"
    elif application_question:
        intent = "application"
    elif any(w in question for w in LIST_WORDS) or GENERIC_CATALOGUE_RE.search(question):
        intent = "list"
    elif fund and understand.wants_details(question):
        intent = "details"
    elif fund and ("คือทุนอะไร" in question or any(w in question for w in FUND_CATEGORY_WORDS)):
        intent = "definition"
    elif any(w in question for w in CEILING_WORDS):
        intent = "ceiling"
    elif any(w in question for w in AMOUNT_WORDS):
        intent = "amount"
    elif any(w in question for w in DEFINITION_WORDS + FUND_CATEGORY_WORDS):
        intent = "definition"
    elif prerequisite or any(w in question for w in ELIGIBILITY_WORDS) or understand.asks_if_possible(question):
        intent = "eligibility"
    elif fund and any(word in question for word in ("อยาก", "ต้องการ", "สนใจ")):
        intent = "match"
    else:
        intent = "unknown"
    if intent == "unknown" and fund and question == fund and previous:
        prior_conflict = _conflicting_clause(previous, understand.mentions(previous))
        if (prior_conflict
                and fund in (prior_conflict["clause_row"]["fund"], prior_conflict["name_row"]["fund"])):
            if any(word in previous for word in DEFINITION_WORDS):
                intent = "definition"
            elif understand.wants_details(previous):
                intent = "details"
    if intent == "unknown" and previous and constrained_followup:
        previous_intent = analyze(previous, history[:-1], profile, use_llm=False)["intent"]
        if previous_intent in ("amount", "ceiling", "tiers", "details", "definition", "eligibility", "application"):
            intent = previous_intent
    if intent == "unknown" and used_context and context.get("intent") in (
            "amount", "ceiling", "tiers", "details", "definition", "eligibility", "application"):
        intent = context["intent"]
    method = "rules"
    allow_slot_llm = use_llm and slot_extractor is not None
    needs_slot_llm = intent == "unknown" or (
        fund is None and intent in ("amount", "ceiling", "tiers", "details", "application")
    )
    if (allow_slot_llm and question and needs_slot_llm
            and not points_back and not broad_eligibility and not personal_history
            and not recommendation_request and not clause_conflict
            and not CLAUSE_REFERENCE_RE.search(question)
            and not chatter and not topic and not open_now and not hypothetical):
        claimed = slot_extractor(question, known_fund=fund)
        if claimed is not None:
            if fund is None and claimed.fund:
                fund, confidence, from_current = claimed.fund, 0.85, True
            if tier is None and claimed.tier:
                tier = claimed.tier
            if assumed and claimed.fiscal_year:
                fiscal_year, assumed = claimed.fiscal_year, False
            if intent == "unknown" and claimed.intent:
                intent = claimed.intent
            method = "rules+llm"
    rewritten = question if from_current else " ".join(x for x in (fund, tier, question) if x)
    slots = {"standalone_question": rewritten.strip(), "fund": fund, "tier": tier, "fiscal_year": fiscal_year,
             "year_assumed": assumed, "intent": intent, "confidence": round(confidence, 2), "method": method,
             "unresolved_reference": points_back and fund is None, "alternatives": alternatives,
             "clause_conflict": clause_conflict,
             "funds": relation_funds if hypothetical else [m["fund"] for m in named],
             "ranking": ranking, "topic": topic, "chatter": chatter,
             "about_asker": understand.about_the_asker(question) and not prerequisite,
             "subject_role": understand.person_asked_about(question), "prerequisite": prerequisite,
             "premise_received": premise_received, "mixed_premise": mixed_premise,
             "catalogue_context": bool(catalogue_context)}
    return slots
