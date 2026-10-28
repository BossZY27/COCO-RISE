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

# Offline research/fixture harness. HTTP requests use QueryService exclusively.
from . import analysis
from .analysis import DEFAULT_YEAR, thai_fiscal_year, _stage

LLM_SLOTS = asyncio.Semaphore(int(os.getenv("RISE_LLM_CONCURRENCY", "1")))
ANSWER_MODE = os.getenv("RISE_ANSWER_MODE", os.getenv("IRFA_ANSWER_MODE", "verified_template")).strip() or "verified_template"
SLOT_LLM_FALLBACK = os.getenv("RISE_SLOT_LLM_FALLBACK", os.getenv("IRFA_SLOT_LLM_FALLBACK", "off")).strip().lower() in ("1", "on", "true", "yes")
SEMANTIC_QUERY_TIMEOUT = max(0.1, float(os.getenv("RISE_SEMANTIC_QUERY_TIMEOUT", "2")))
SEMANTIC_RETRY_SECONDS = max(1.0, float(os.getenv("RISE_SEMANTIC_RETRY_SECONDS", "30")))
_SEMANTIC_CIRCUIT_LOCK = threading.Lock()
_SEMANTIC_FAILURE_UNTIL = 0.0

def analyze(question, history=None, profile=None, context=None, use_llm=None):
    return analysis.analyze(question, history, profile, context,
                            use_llm=SLOT_LLM_FALLBACK if use_llm is None else use_llm,
                            slot_extractor=claims.extract_slots_with_llm)


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
