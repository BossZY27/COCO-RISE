"""The four online steps from the architecture figure.

1 analyse the question      rules first, LLM only when the rules cannot fill the slots
2 retrieve evidence         keyword, semantic and structured lookup in parallel, then RRF
3 build the answer          eligibility from rules and profile, wording from a template or an LLM
4 verify then answer        every number and citation must appear in the evidence

Each step records its own time so the latency budget can be checked.
"""
import asyncio
import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo

from app.modules.knowledge import corpus as store
from app.infrastructure.ollama import legacy as llm
from . import persona as identity
from . import answers, claims, understand
from .composer import _catalogue_answer, _pending_notice, compose_answer
from .patterns import (
    BROAD_ELIGIBILITY_RE,
    GENERIC_CATALOGUE_RE,
    PERSONAL_HISTORY_RE,
    REPEAT_ELIGIBILITY_RE,
)
from app.modules.knowledge.formatting import baht as _baht, cite as _cite
from .verification import verify

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
# One question at a time reaches the model. A second caller waits rather than halving the speed of both,
# because on a CPU two answers in parallel take about twice as long each and both miss the 40 second budget.
LLM_SLOTS = asyncio.Semaphore(int(os.getenv("RISE_LLM_CONCURRENCY", "1")))
_DEFAULT_ANSWER_MODE = "verified_template" if "pytest" in sys.modules else "llm"
ANSWER_MODE = os.getenv("RISE_ANSWER_MODE", os.getenv("IRFA_ANSWER_MODE", _DEFAULT_ANSWER_MODE)).strip() or _DEFAULT_ANSWER_MODE
_DEFAULT_SLOT_FALLBACK = "off" if "pytest" in sys.modules else "on"
SLOT_LLM_FALLBACK = os.getenv("RISE_SLOT_LLM_FALLBACK", os.getenv("IRFA_SLOT_LLM_FALLBACK", _DEFAULT_SLOT_FALLBACK)).strip().lower() in ("1", "on", "true", "yes")
SEMANTIC_QUERY_TIMEOUT = max(0.1, float(os.getenv("RISE_SEMANTIC_QUERY_TIMEOUT", "2")))
SEMANTIC_RETRY_SECONDS = max(1.0, float(os.getenv("RISE_SEMANTIC_RETRY_SECONDS", "30")))
_SEMANTIC_CIRCUIT_LOCK = threading.Lock()
_SEMANTIC_FAILURE_UNTIL = 0.0
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
TEMPLATE_ONLY_INTENTS = ("compare", "rank", "tiers", "details", "catalogue_details", "hypothetical", "definition",
                         "eligibility", "application", "unknown", "match", "open_now", "recommendation")
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
CLAUSE_REFERENCE_RE = re.compile(r"ข้อ\s*(?:ที่\s*)?([12]\.\d{1,2})(?![\d.])")


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
            context: dict | None = None, use_llm: bool | None = None) -> dict:
    """Fill the slots the rest of the pipeline needs.

    A follow-up such as "แล้วชั้นกลางล่ะ" only makes sense with the previous turn, so missing slots fall back
    to the last question. What the current question says always wins. An LLM is called only when the rules
    cannot find the fund at all.
    """
    question = understand.normalize(question)
    history = [understand.normalize(turn) for turn in history or []]
    previous = history[-1] if history else ""
    context = context if isinstance(context, dict) else {}
    context_fund = context.get("fund") if context.get("fund") in store.FUNDS else None
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
    allow_slot_llm = use_llm if use_llm is not None else SLOT_LLM_FALLBACK
    needs_slot_llm = intent == "unknown" or (
        fund is None and intent in ("amount", "ceiling", "tiers", "details", "application")
    )
    if (allow_slot_llm and question and needs_slot_llm
            and not points_back and not broad_eligibility and not personal_history
            and not recommendation_request and not clause_conflict
            and not CLAUSE_REFERENCE_RE.search(question)
            and not chatter and not topic and not open_now and not hypothetical):
        claimed = claims.extract_slots_with_llm(question, known_fund=fund)
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


# ---------------------------------------------------------------- step 2: retrieve
def _semantic(query: str, year: int | None) -> tuple[list[dict], str]:
    """Real vectors when the embedding model answers, trigrams when it does not."""
    global _SEMANTIC_FAILURE_UNTIL
    if ANSWER_MODE in ("llm", "typed_claim"):
        # On CPU-only Ollama, calling bge-m3 unloads the active chat LLM and adds ~10s reload latency.
        return store.semantic_search(query, year, 10), "trigram"
    if store.vectors_ready():
        # A complete on-disk cache can still be present while Ollama is down.
        # Without a circuit breaker every question would pay the transport
        # timeout before reaching the trigram fallback.
        with _SEMANTIC_CIRCUIT_LOCK:
            if time.monotonic() >= _SEMANTIC_FAILURE_UNTIL:
                try:
                    vector = llm.embed([query], timeout=SEMANTIC_QUERY_TIMEOUT)[0]
                    hits = store.vector_search(vector, year, 10)
                    _SEMANTIC_FAILURE_UNTIL = 0.0
                    return hits, llm.EMBED_MODEL
                except Exception:
                    _SEMANTIC_FAILURE_UNTIL = time.monotonic() + SEMANTIC_RETRY_SECONDS
    return store.semantic_search(query, year, 10), "trigram"


def semantic_backend() -> str:
    """Report the retrieval path a request would use at this moment."""
    if not store.vectors_ready() or time.monotonic() < _SEMANTIC_FAILURE_UNTIL:
        return "trigram"
    return llm.EMBED_MODEL


async def retrieve(slots: dict) -> dict:
    query = slots["standalone_question"]
    year = slots["fiscal_year"]
    if slots["intent"] in ("list", "catalogue_details", "open_now", "recommendation", "clarify_reference"):
        # The catalogue answer and citations are built directly from the verified
        # structured rows. Searching passages or embedding the query cannot change it.
        return {"chunks": [], "facts": [], "semantic_by": "not_needed",
                "ceiling_fallback": False, "pending_review": False,
                "keyword_ids": [], "semantic_ids": []}
    amount_type = "ceiling" if slots["intent"] == "ceiling" else "allocated"
    keyword, (semantic, how), hierarchy, facts = await asyncio.gather(
        asyncio.to_thread(store.keyword_search, query, year, 10),
        asyncio.to_thread(_semantic, query, year),
        asyncio.to_thread(store.hierarchical_search, query, year, 10),
        asyncio.to_thread(store.fact_lookup, slots["fund"], slots["tier"], year, amount_type),
    )
    # OCR-derived structured rows are just as untrusted as draft passages. Keep them visible
    # in /admin, but never put their figures into the answer or the model's fact block.
    pending_review = any(not store.is_verified(f) for f in facts)
    facts = [f for f in facts if store.is_verified(f)]
    fell_back = False
    if not facts and amount_type == "allocated" and not pending_review and slots["intent"] in ("amount", "eligibility"):
        # No announcement for that year, but the central criteria still says what the ceiling is.
        # Giving the ceiling and saying so beats answering "ข้อมูลไม่เพียงพอ" while holding the number.
        ceiling_rows = store.fact_lookup(slots["fund"], slots["tier"], None, "ceiling")
        pending_review = pending_review or any(not store.is_verified(f) for f in ceiling_rows)
        facts = [f for f in ceiling_rows if store.is_verified(f)]
        fell_back = bool(facts)
    return {"chunks": (store.semantic_guarded_rrf(keyword, semantic, hierarchy)
                       if how == llm.EMBED_MODEL else store.rrf([keyword, semantic, hierarchy])),
            "facts": facts, "semantic_by": how,
            "ceiling_fallback": fell_back, "pending_review": pending_review,
            "keyword_ids": [c["id"] for c in keyword], "semantic_ids": [c["id"] for c in semantic]}


# ---------------------------------------------------------------- step 3: eligibility and answer
def _profile_value(profile: dict, field: str):
    if field.startswith("grants_this_year."):
        return (profile.get("grants_this_year") or {}).get(field.split(".", 1)[1], 0)
    return profile.get(field)


def _blocking_grant(profile: dict, cond: dict) -> str | None:
    """Cross-fund exclusion: name the grant the person already holds that rules this one out.

    The central criteria states one of these outright, that a person paid the award under clause 2.2
    cannot claim the one under 2.3. More of them live in the annual announcements, which are scans,
    so the team adds those rows as each announcement is read.
    """
    held = profile.get("grants_this_year") or {}
    return next((fund for fund in cond["value"] if held.get(fund, 0) > 0), None)


def check_rules(profile: dict, slots: dict) -> dict:
    rules = store.rules_for(slots["fund"], slots["fiscal_year"])
    if not rules:
        return {"verdict": "no_rule", "reason": None, "source": None, "missing": []}
    rule = rules[0]
    missing = [c["field"] for c in rule["conditions"]
               if _profile_value(profile, c["field"]) is None and not c["field"].startswith("grants_this_year.")]
    if missing:
        return {"verdict": "unknown", "reason": None, "source": rule["source"], "missing": missing}
    for cond in rule["conditions"]:
        value = _profile_value(profile, cond["field"])
        blocker = _blocking_grant(profile, cond) if cond["op"] == "not_received" else None
        ok = {"in": lambda: value in cond["value"],
              "eq": lambda: value == cond["value"],
              "lt": lambda: (value or 0) < cond["value"],
              "not_received": lambda: blocker is None}[cond["op"]]()
        if not ok:
            return {"verdict": "not_eligible", "reason": cond["fail_reason"].replace("{fund}", blocker or ""),
                    "source": rule["source"], "missing": [], "field": cond["field"]}
    return {"verdict": "eligible", "reason": None, "source": rule["source"], "missing": []}


def _what_if_ruling(profile: dict, slots: dict) -> dict:
    """"ถ้าได้ A แล้ว ขอ B ได้ไหม": B's rules against the asker's record with A added to this year's grants.
    The record is not changed; the copy lives only for this answer."""
    if slots.get("mixed_premise"):
        return {"verdict": "not_evaluated", "reason": None, "source": None, "missing": []}
    *received, target = slots["funds"]
    held = dict(profile.get("grants_this_year") or {})
    if slots.get("premise_received", True):
        for fund in received:
            held[fund] = held.get(fund, 0) + 1
    else:
        for fund in received:
            held.pop(fund, None)
    hyp_profile = {**profile, "grants_this_year": held}
    ruling = check_rules(hyp_profile, {**slots, "fund": target})
    if ruling["verdict"] == "unknown" and slots.get("premise_received", True):
        # Allow anonymous users (P5) to check cross-grant restrictions without disclosing personal attributes.
        rules = store.rules_for(target, slots["fiscal_year"])
        if rules:
            rule = rules[0]
            for cond in rule["conditions"]:
                if cond["op"] == "not_received":
                    blocker = _blocking_grant(hyp_profile, cond)
                    if blocker is not None:
                        return {
                            "verdict": "not_eligible",
                            "reason": cond["fail_reason"].replace("{fund}", blocker),
                            "source": rule["source"],
                            "missing": [],
                            "field": cond["field"],
                        }
                elif cond["op"] == "lt" and cond["field"].startswith("grants_this_year."):
                    value = _profile_value(hyp_profile, cond["field"])
                    if not ((value or 0) < cond["value"]):
                        return {
                            "verdict": "not_eligible",
                            "reason": cond["fail_reason"],
                            "source": rule["source"],
                            "missing": [],
                            "field": cond["field"],
                        }
    return ruling




# ---------------------------------------------------------------- step 3b: natural wording from the local model
SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยตอบคำถามทุนวิจัยของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น\n"
    "เรียบเรียงคำตอบภาษาไทยจากข้อมูลที่ตรวจแล้วด้านล่างเท่านั้น\n"
    "กติกา: ห้ามเพิ่มหรือแก้ตัวเลขใด ๆ ห้ามเดาข้อมูลที่ไม่มีในข้อมูลที่ให้ "
    "ตอบสั้นที่สุด 1 ถึง 2 ประโยค ไม่ต้องใส่รายการอ้างอิงเอง ระบบจะเติมให้ "
    "ถ้าข้อมูลที่ให้ไม่พอ ให้ตอบว่า ข้อมูลไม่เพียงพอ")


def fact_block(slots: dict, used: list[dict], ruling: dict, citations: list[str],
               ceiling_fallback: bool = False, pending_review: bool = False) -> str:
    """The only content the model is allowed to use. Numbers come from the structured layer, not from the model."""
    fact = used[0]
    lines = [f"ทุน: {fact['fund']}{(' ' + fact['tier']) if fact['tier'] else ''}"]
    if fact["amount_type"] == "ceiling":
        if pending_review:
            lines.append(f"ประกาศจัดสรรปีงบประมาณ {slots['fiscal_year']} ยังรอเจ้าหน้าที่ตรวจและอนุมัติ "
                         "จึงยังตอบวงเงินจัดสรรรายปีไม่ได้ ต้องบอกผู้ถามด้วย")
        if ceiling_fallback:
            lines.append(f"หมายเหตุสำคัญ: ระบบยังไม่มีข้อมูลวงเงินจัดสรรของปีงบประมาณ {slots['fiscal_year']} "
                         "ตัวเลขข้างล่างจึงเป็นเพดานของหลักเกณฑ์กลาง ต้องบอกผู้ถามด้วย")
        if fact["amount"]:
            lines.append(f"เพดานตามหลักเกณฑ์กลาง: {_baht(fact['amount'])} บาทต่อทุน")
        else:
            lines.append("เพดานต่อทุน: หลักเกณฑ์กลางไม่ได้กำหนดไว้ กำหนดเฉพาะวงเงินรวม")
        if fact.get("max_grants_per_year"):
            lines.append(f"จำนวนทุนต่อปี: ไม่เกิน {fact['max_grants_per_year']} ทุน")
        lines.append("หมายเหตุ: เพดานไม่ใช่วงเงินที่จัดสรรจริงของปีใดปีหนึ่ง")
    else:
        lines.append(f"ปีงบประมาณ: {fact['fiscal_year']}")
        # Say what to write, not only what to avoid. An earlier version said "ห้ามเขียนว่าได้เท่านี้"
        # and the model answered by leaving the figure out altogether, which helps nobody.
        lines.append(f"วงเงินสูงสุดที่จัดสรรปีนี้: {_baht(fact['amount'])} บาทต่อทุน "
                     f"(ต้องบอกตัวเลข {_baht(fact['amount'])} ในคำตอบ และเขียนว่าสูงสุดหรือไม่เกิน)")
        if fact["max_grants_per_person"]:
            lines.append(f"ขอได้คนละไม่เกิน: {fact['max_grants_per_person']} ทุนต่อปี")
        if fact.get("max_grants_per_year"):
            lines.append(f"จำนวนทุนรวมในประกาศปีนี้: ไม่เกิน {fact['max_grants_per_year']} ทุน")
        if len(used) > 1:
            lines.append(f"เพดานตามหลักเกณฑ์กลาง: {_baht(used[1]['amount'])} บาท "
                         "(คนละระดับกับวงเงินจริง ห้ามนำมาตอบแทนกัน)")
    if fact["total_cap"]:
        lines.append(f"วงเงินรวมไม่เกิน: {_baht(fact['total_cap'])} บาท")
    if fact.get("note"):
        lines.append(f"หมายเหตุอัตรา: {fact['note']}")
    if ruling["verdict"] in ("eligible", "not_eligible", "unknown"):
        verdict_text = {
            "eligible": "เข้าเกณฑ์",
            "not_eligible": f"ยังไม่เข้าเกณฑ์ เพราะ{ruling['reason']}",
            "unknown": "ยังสรุปไม่ได้ เพราะระบบไม่มีข้อมูลประเภทบุคลากรและสังกัดของผู้ถาม",
        }[ruling["verdict"]]
        lines.append(f"สิทธิ์ของผู้ถาม: {verdict_text}")
    lines.append(f"อ้างอิง: {' | '.join(citations)}")
    return "\n".join(lines)


def llm_messages(question: str, block: str) -> list[dict]:
    # The model only rewrites reviewed facts. It never needs the raw user message,
    # which may contain instructions intended to override this boundary.
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"ข้อมูลที่ตรวจแล้ว:\n{block}\n\nเรียบเรียงข้อมูลนี้เท่านั้น"}]




# ---------------------------------------------------------------- orchestration
def _result(answer, citations, slots, ruling, evidence, check, times, profile, source, selected_model):
    if not _current_corpus():
        answer, citations, source = "หลักฐานมีการเปลี่ยนแปลง กรุณาถามใหม่", [], "blocked"
        check = {"ok": False, "problems": ["Corpus changed while answering"]}
    times["total"] = round(sum(v for k, v in times.items() if k != "total" and isinstance(v, (int, float))), 1)
    return {"answer": answer, "citations": citations, "slots": slots, "ruling": ruling, "answer_source": source,
            "selected_model": selected_model,
            "evidence": {"chunks": [c["id"] for c in evidence["chunks"]],
                         "facts": [f["id"] for f in evidence["facts"]],
                         "keyword": evidence["keyword_ids"], "semantic": evidence["semantic_ids"],
                         "semantic_by": evidence["semantic_by"]},
            "verify": check, "timings_ms": times,
            "profile": {"id": profile["id"], "name": profile["name"], "source": profile["source"]}}


@contextmanager
def _queue_wait(times: dict):
    """Count the time spent waiting for the model to be free, so a slow answer can be explained."""
    t0 = time.perf_counter()
    yield
    waited = round((time.perf_counter() - t0) * 1000, 1)
    if waited > 50:
        times["llm_wait"] = waited


async def _prepare(question: str, persona_id: str, history: list[str] | None, times: dict,
                   context: dict | None = None):
    """Steps 1 to 3: everything that happens before the model starts writing."""
    # An unknown asker still gets an answer, but the eligibility line says the system cannot judge yet.
    profile = identity.lookup(persona_id) or store.view().PERSONAS["P5"]
    with _stage(times, "analyze"):
        slots = analyze(question, history, profile, context)
    with _stage(times, "retrieve"):
        evidence = await retrieve(slots)
    with _stage(times, "rules"):
        if slots["intent"] == "hypothetical":
            ruling = _what_if_ruling(profile, slots)
        elif not slots.get("about_asker", True):
            ruling = {"verdict": "not_evaluated", "reason": None, "source": None, "missing": []}
        else:
            ruling = check_rules(profile, slots)
    with _stage(times, "answer"):
        template, citations, used = compose_answer(slots, evidence, ruling, profile)
        before, after = answers.name_notes(slots)
        template = " ".join(part for part in (before, template, after) if part)
    return profile, slots, evidence, ruling, template, citations, used


def _model_may_rephrase(used: list[dict], evidence: dict, slots: dict | None = None) -> bool:
    """Free-form model prose is fail-closed until it can emit typed, validated claims.

    Numeric and citation checks cannot prove arbitrary Thai prose is entailed by a
    source. The code-built template is therefore the only fact-bearing answer that
    may leave the pipeline; a future model path must return closed-schema claims.
    """
    return False


def _model_may_generate_rag(
    used: list[dict],
    citations: list[str],
    evidence: dict,
    slots: dict | None = None,
    model: str | None = None,
) -> bool:
    """Return True when Step 3 LLM Answer Generator may write natural Thai prose from verified RAG evidence."""
    return claims.may_generate_llm_answer(used, citations, evidence, slots, ANSWER_MODE, model)


def _model_may_emit_claims(
    used: list[dict],
    evidence: dict,
    slots: dict | None = None,
    model: str | None = None,
) -> bool:
    """Return True when closed-schema typed claims may be requested and validated."""
    return claims.may_emit_typed_claims(used, evidence, slots, ANSWER_MODE, model)


def _model_may_emit_passage_claim(
    used: list[dict],
    citations: list[str],
    evidence: dict,
    slots: dict | None = None,
    model: str | None = None,
) -> bool:
    """Return True when a verified non-fact answer may emit a TypedPassageClaim."""
    return claims.may_emit_passage_claim(used, citations, evidence, slots, ANSWER_MODE, model)


async def run(question: str, persona_id: str = "P1", history: list[str] | None = None,
              model: str | None = None, context: dict | None = None) -> dict:
    times: dict[str, float] = {}
    selected_model = None
    profile, slots, evidence, ruling, answer, citations, used = await _prepare(
        question, persona_id, history, times, context)
    source = "template"
    if ANSWER_MODE == "llm" and _model_may_generate_rag(used, citations, evidence, slots, model):
        effective_model = model or llm.SMALL or llm.MAIN
        with _queue_wait(times):
            await LLM_SLOTS.acquire()
        try:
            with _stage(times, "llm"):
                ok, rag_problems, llm_text, chosen_model = await asyncio.to_thread(
                    claims.request_llm_rag_answer, question, slots, used, ruling, evidence, answer, effective_model
                )
                if ok and llm_text:
                    vcheck = verify(llm_text, citations, used, evidence, slots, slots["intent"] != "list", ruling)
                    if vcheck["ok"]:
                        answer = llm_text
                        selected_model = chosen_model
                        source = f"llm:{chosen_model}"
                    else:
                        times["claim_rejected"] = "; ".join(vcheck["problems"][:2]) or "verify_failed"
                else:
                    times["claim_rejected"] = "; ".join(rag_problems[:2]) if rag_problems else "invalid_prose"
        finally:
            LLM_SLOTS.release()
    elif _model_may_emit_claims(used, evidence, slots, model):
        effective_model = model or llm.SMALL or llm.MAIN
        with _queue_wait(times):
            await LLM_SLOTS.acquire()
        try:
            with _stage(times, "llm"):
                ok, claim_problems, bundle, chosen_model = await asyncio.to_thread(
                    claims.request_typed_claims, slots, used, ruling, evidence, effective_model
                )
                if ok and bundle is not None:
                    rendered, rendered_citations, rendered_used = claims.render_typed_claims(
                        bundle, slots, evidence, ruling, profile
                    )
                    if verify(rendered, rendered_citations, rendered_used, evidence, slots, ruling=ruling)["ok"]:
                        answer, citations, used = rendered, rendered_citations, rendered_used
                        selected_model = chosen_model
                        source = f"typed_claim:{chosen_model}"
                    else:
                        times["claim_rejected"] = "verify_failed"
                else:
                    times["claim_rejected"] = "; ".join(claim_problems[:2]) if claim_problems else "invalid_claim"
        finally:
            LLM_SLOTS.release()
    elif _model_may_emit_passage_claim(used, citations, evidence, slots, model):
        effective_model = model or llm.SMALL or llm.MAIN
        with _queue_wait(times):
            await LLM_SLOTS.acquire()
        try:
            with _stage(times, "llm"):
                ok, claim_problems, pclaim, chosen_model = await asyncio.to_thread(
                    claims.request_passage_claim, slots, citations, effective_model
                )
                if ok and pclaim is not None:
                    if verify(answer, citations, used, evidence, slots, slots["intent"] != "list", ruling)["ok"]:
                        selected_model = chosen_model
                        source = f"typed_claim:{chosen_model}"
                    else:
                        times["claim_rejected"] = "verify_failed"
                else:
                    times["claim_rejected"] = "; ".join(claim_problems[:2]) if claim_problems else "invalid_claim"
        finally:
            LLM_SLOTS.release()
    elif _model_may_rephrase(used, evidence, slots):
        with _queue_wait(times):
            await LLM_SLOTS.acquire()
        try:
            with _stage(times, "llm"):
                try:
                    block = fact_block(slots, used, ruling, citations, evidence.get("ceiling_fallback", False),
                                       evidence.get("pending_review", False))
                    text = await asyncio.to_thread(llm.chat, llm_messages(question, block), selected_model)
                    if text and verify(text, citations, used, evidence, slots, ruling=ruling)["ok"]:
                        answer, source = text, selected_model      # the model only rephrases checked facts
                except Exception as exc:
                    times["llm_error"] = type(exc).__name__
        finally:
            LLM_SLOTS.release()
    with _stage(times, "verify"):
        check = verify(answer, citations, used, evidence, slots, slots["intent"] != "list", ruling)
    if not check["ok"]:
        answer = "ข้อมูลไม่เพียงพอ ระบบตรวจแล้วพบว่าคำตอบยังไม่ตรงกับหลักฐาน จึงไม่แสดงคำตอบนี้"
        citations, source = [], "blocked"
    return _result(answer, citations, slots, ruling, evidence, check, times, profile, source, selected_model)


async def _to_async(make_generator):
    """Run a blocking generator in a thread and hand its pieces to the event loop."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker():
        try:
            for item in make_generator():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = await queue.get()
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item


async def run_stream(question: str, persona_id: str = "P1", history: list[str] | None = None,
                     model: str | None = None, context: dict | None = None):
    """Send a verified template immediately, then a verified final answer after model generation.

    Raw model tokens must never reach the client before the final gate has checked them.
    """
    times: dict[str, float] = {}
    selected_model = None
    profile, slots, evidence, ruling, template, citations, used = await _prepare(
        question, persona_id, history, times, context)
    yield {"type": "stage", "slots": slots, "ruling": ruling, "timings_ms": dict(times),
           "semantic_by": evidence["semantic_by"]}

    answer, source = template, "template"
    if _current_corpus() and verify(template, citations, used, evidence, slots, slots["intent"] != "list", ruling)["ok"]:
        yield {"type": "preview", "answer": template, "citations": citations}
    if ANSWER_MODE == "llm" and _model_may_generate_rag(used, citations, evidence, slots, model):
        effective_model = model or llm.SMALL or llm.MAIN
        if LLM_SLOTS.locked():
            yield {"type": "stage", "waiting": "มีคำถามอื่นใช้โมเดลอยู่ กำลังรอคิว"}
        with _queue_wait(times):
            await LLM_SLOTS.acquire()
        t0 = time.perf_counter()
        try:
            ok, rag_problems, llm_text, chosen_model = await asyncio.to_thread(
                claims.request_llm_rag_answer, question, slots, used, ruling, evidence, template, effective_model
            )
            times["llm"] = round((time.perf_counter() - t0) * 1000, 1)
            if ok and llm_text:
                vcheck = verify(llm_text, citations, used, evidence, slots, slots["intent"] != "list", ruling)
                if vcheck["ok"]:
                    answer = llm_text
                    selected_model = chosen_model
                    source = f"llm:{chosen_model}"
                else:
                    times["claim_rejected"] = "; ".join(vcheck["problems"][:2]) or "verify_failed"
                    yield {"type": "replace", "reason": "คำตอบจากโมเดลไม่ผ่านการตรวจ จึงใช้คำตอบจากข้อมูลที่ตรวจแล้ว"}
            else:
                times["claim_rejected"] = "; ".join(rag_problems[:2]) if rag_problems else "invalid_prose"
                yield {"type": "replace", "reason": "คำตอบจากโมเดลไม่ผ่านการตรวจ จึงใช้คำตอบจากข้อมูลที่ตรวจแล้ว"}
        except Exception as exc:
            times["llm"] = round((time.perf_counter() - t0) * 1000, 1)
            times["llm_error"] = type(exc).__name__
            yield {"type": "replace", "reason": "เรียกโมเดลไม่สำเร็จ จึงใช้คำตอบจากข้อมูลที่ตรวจแล้ว"}
        finally:
            LLM_SLOTS.release()
    elif _model_may_emit_claims(used, evidence, slots, model):
        effective_model = model or llm.SMALL or llm.MAIN
        if LLM_SLOTS.locked():
            yield {"type": "stage", "waiting": "มีคำถามอื่นใช้โมเดลอยู่ กำลังรอคิว"}
        with _queue_wait(times):
            await LLM_SLOTS.acquire()
        t0 = time.perf_counter()
        try:
            ok, claim_problems, bundle, chosen_model = await asyncio.to_thread(
                claims.request_typed_claims, slots, used, ruling, evidence, effective_model
            )
            times["llm"] = round((time.perf_counter() - t0) * 1000, 1)
            if ok and bundle is not None:
                rendered, rendered_citations, rendered_used = claims.render_typed_claims(
                    bundle, slots, evidence, ruling, profile
                )
                if verify(rendered, rendered_citations, rendered_used, evidence, slots, ruling=ruling)["ok"]:
                    answer, citations, used = rendered, rendered_citations, rendered_used
                    selected_model = chosen_model
                    source = f"typed_claim:{chosen_model}"
                else:
                    times["claim_rejected"] = "verify_failed"
                    yield {"type": "replace", "reason": "คำตอบจากโมเดลไม่ผ่านการตรวจ จึงใช้คำตอบจากข้อมูลที่ตรวจแล้ว"}
            else:
                times["claim_rejected"] = "; ".join(claim_problems[:2]) if claim_problems else "invalid_claim"
                yield {"type": "replace", "reason": "คำตอบจากโมเดลไม่ผ่านการตรวจ จึงใช้คำตอบจากข้อมูลที่ตรวจแล้ว"}
        except Exception as exc:
            times["llm"] = round((time.perf_counter() - t0) * 1000, 1)
            times["llm_error"] = type(exc).__name__
            yield {"type": "replace", "reason": "เรียกโมเดลไม่สำเร็จ จึงใช้คำตอบจากข้อมูลที่ตรวจแล้ว"}
        finally:
            LLM_SLOTS.release()
    elif _model_may_emit_passage_claim(used, citations, evidence, slots, model):
        effective_model = model or llm.SMALL or llm.MAIN
        if LLM_SLOTS.locked():
            yield {"type": "stage", "waiting": "มีคำถามอื่นใช้โมเดลอยู่ กำลังรอคิว"}
        with _queue_wait(times):
            await LLM_SLOTS.acquire()
        t0 = time.perf_counter()
        try:
            ok, claim_problems, pclaim, chosen_model = await asyncio.to_thread(
                claims.request_passage_claim, slots, citations, effective_model
            )
            times["llm"] = round((time.perf_counter() - t0) * 1000, 1)
            if ok and pclaim is not None:
                if verify(answer, citations, used, evidence, slots, slots["intent"] != "list", ruling)["ok"]:
                    selected_model = chosen_model
                    source = f"typed_claim:{chosen_model}"
                else:
                    times["claim_rejected"] = "verify_failed"
            else:
                times["claim_rejected"] = "; ".join(claim_problems[:2]) if claim_problems else "invalid_claim"
        except Exception as exc:
            times["llm"] = round((time.perf_counter() - t0) * 1000, 1)
            times["llm_error"] = type(exc).__name__
        finally:
            LLM_SLOTS.release()
    elif _model_may_rephrase(used, evidence, slots):
        if LLM_SLOTS.locked():
            yield {"type": "stage", "waiting": "มีคำถามอื่นใช้โมเดลอยู่ กำลังรอคิว"}
        with _queue_wait(times):
            await LLM_SLOTS.acquire()
        block = fact_block(slots, used, ruling, citations, evidence.get("ceiling_fallback", False),
                           evidence.get("pending_review", False))
        pieces: list[str] = []
        t0 = time.perf_counter()
        try:
            async for piece in _to_async(lambda: llm.stream_chat(llm_messages(question, block), selected_model)):
                pieces.append(piece)
            times["llm"] = round((time.perf_counter() - t0) * 1000, 1)
            text = llm.THINK_TAG.sub("", "".join(pieces)).strip() if pieces else ""
            if text and verify(text, citations, used, evidence, slots, ruling=ruling)["ok"]:
                answer, source = text, selected_model
            else:
                yield {"type": "replace", "reason": "คำตอบจากโมเดลไม่ผ่านการตรวจ จึงใช้คำตอบจากข้อมูลที่ตรวจแล้ว"}
        except Exception as exc:
            times["llm"] = round((time.perf_counter() - t0) * 1000, 1)
            times["llm_error"] = type(exc).__name__
            yield {"type": "replace", "reason": "เรียกโมเดลไม่สำเร็จ จึงใช้คำตอบจากข้อมูลที่ตรวจแล้ว"}
        finally:
            LLM_SLOTS.release()

    with _stage(times, "verify"):
        check = verify(answer, citations, used, evidence, slots, slots["intent"] != "list", ruling)
    if not check["ok"]:
        answer, citations, source = ("ข้อมูลไม่เพียงพอ ระบบตรวจแล้วพบว่าคำตอบยังไม่ตรงกับหลักฐาน "
                                     "จึงไม่แสดงคำตอบนี้"), [], "blocked"
    yield {"type": "done", **_result(answer, citations, slots, ruling, evidence, check, times, profile,
                                      source, selected_model)}


def _current_corpus():
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    return runtime is None or runtime.repository.corpus_revision() == store.view().revision
