"""Closed-schema typed claims and slot extraction for safe LLM participation.

Option 1 architecture:
1. Step 1 (Query Understanding): Rule-based extraction runs first. When rules leave the
   intent unresolved on an unguarded query and LLM slot extraction is enabled, the model
   may return a closed-schema ``TypedSlotClaim`` JSON object that code validates against
   ``store.FUNDS``, ``store.TIERS``, and lexical anchors in the question.
2. Step 3 (Typed-Claim Selection & Deterministic Rendering): Instead of writing free-form
   Thai prose, the model returns a closed-schema ``TypedClaimBundle`` JSON object selected
   from verified ``used`` facts and ``ruling``. Code validates every field 1:1 against the
   verified evidence (source document/clause/page, ``amount_type``, ``fiscal_year``,
   ``amount``, ``tier``, ``total_cap``, grant counts, rate ``note``, and ``verdict``) and
   renders the final Thai answer deterministically before ``verification.verify`` runs.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app.core import security as limits
from app.infrastructure.ollama import legacy as llm
from app.modules.knowledge import corpus as store
from . import answers
from .composer import compose_answer

ALLOWED_SLOT_INTENTS = (
    "amount",
    "ceiling",
    "tiers",
    "details",
    "definition",
    "eligibility",
    "application",
    "list",
    "catalogue_details",
)

SlotIntentLiteral = Literal[
    "amount",
    "ceiling",
    "tiers",
    "details",
    "definition",
    "eligibility",
    "application",
    "list",
    "catalogue_details",
]

ClaimIntentLiteral = Literal["amount", "ceiling", "eligibility", "tiers", "details"]
AmountTypeLiteral = Literal["ceiling", "allocated"]
VerdictLiteral = Literal["eligible", "not_eligible", "unknown", "no_rule", "not_evaluated"]

CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.S)
YEAR_TOKEN_RE = re.compile(r"(?<!\d)(25\d\d)(?!\d)")

CLAIM_TIMEOUT = max(0.5, float(os.getenv("RISE_CLAIM_TIMEOUT", os.getenv("IRFA_CLAIM_TIMEOUT", "28"))))
SLOT_TIMEOUT = max(0.5, float(os.getenv("RISE_SLOT_TIMEOUT", os.getenv("IRFA_SLOT_TIMEOUT", "15"))))
CLAIM_RETRY_SECONDS = max(
    1.0,
    float(os.getenv("RISE_CLAIM_RETRY_SECONDS", os.getenv("IRFA_CLAIM_RETRY_SECONDS", "5"))),
)
_CLAIM_CIRCUIT_LOCK = threading.Lock()
_CLAIM_FAILURE_UNTIL = 0.0


class TypedSlotClaim(BaseModel):
    """Closed schema for Step 1 LLM slot-filling fallback."""

    model_config = ConfigDict(extra="forbid")

    fund: str | None = None
    tier: str | None = None
    fiscal_year: int | None = None
    intent: SlotIntentLiteral | None = None


class TypedFactClaim(BaseModel):
    """Closed schema for a single verified fact claim in Step 3."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str | None = None
    fund: str
    tier: str | None = None
    amount_type: AmountTypeLiteral
    fiscal_year: int | None = None
    amount: int | None = None
    unit: Literal["บาทต่อทุน", "บาทต่อเรื่อง", "บาท"] = "บาทต่อทุน"
    total_cap: int | None = None
    max_grants_per_person: int | None = None
    max_grants_per_year: int | None = None
    note: str | None = None
    source_doc: str
    source_clause: str
    source_page: int | None = None


class TypedClaimBundle(BaseModel):
    """Closed schema for the complete structured claim set returned by the LLM."""

    model_config = ConfigDict(extra="forbid")

    intent: ClaimIntentLiteral
    claims: list[TypedFactClaim]
    verdict: VerdictLiteral = "no_rule"
    verdict_reason: str | None = None
    verdict_source_doc: str | None = None
    verdict_source_clause: str | None = None
    verdict_source_page: int | None = None
    year_assumed: bool = False


class TypedPassageClaim(BaseModel):
    """Closed schema for verified passage, definition, or catalogue claims when used_facts is empty."""

    model_config = ConfigDict(extra="forbid")

    intent: str
    fund: str | None = None
    citation_count: int
    primary_citation: str


def _clean_json_text(raw_text: str) -> str:
    cleaned = llm.THINK_TAG.sub("", raw_text or "").strip()
    cleaned = CODE_FENCE_RE.sub("", cleaned).strip()
    return cleaned


def bundle_from_verified(slots: dict, used: list[dict], ruling: dict) -> TypedClaimBundle:
    """Build the canonical TypedClaimBundle from verified facts and ruling."""
    allowed_intents = ("amount", "ceiling", "eligibility", "tiers", "details")
    intent = slots.get("intent") if slots.get("intent") in allowed_intents else "amount"
    claims: list[TypedFactClaim] = []
    for fact in used:
        source = fact.get("source") or {}
        claims.append(
            TypedFactClaim(
                fact_id=fact.get("id"),
                fund=fact["fund"],
                tier=fact.get("tier"),
                amount_type=fact["amount_type"],
                fiscal_year=fact.get("fiscal_year"),
                amount=fact.get("amount"),
                unit="บาทต่อทุน",
                total_cap=fact.get("total_cap"),
                max_grants_per_person=fact.get("max_grants_per_person"),
                max_grants_per_year=fact.get("max_grants_per_year"),
                note=fact.get("note"),
                source_doc=source.get("doc", ""),
                source_clause=source.get("clause", ""),
                source_page=source.get("page"),
            )
        )
    ruling_source = (ruling or {}).get("source") or {}
    verdict = (ruling or {}).get("verdict") or "no_rule"
    if verdict not in ("eligible", "not_eligible", "unknown", "no_rule", "not_evaluated"):
        verdict = "no_rule"
    return TypedClaimBundle(
        intent=intent,
        claims=claims,
        verdict=verdict,
        verdict_reason=(ruling or {}).get("reason"),
        verdict_source_doc=ruling_source.get("doc"),
        verdict_source_clause=ruling_source.get("clause"),
        verdict_source_page=ruling_source.get("page"),
        year_assumed=bool(slots.get("year_assumed", False)),
    )


CLAIM_SYSTEM_PROMPT = (
    "คุณคือโมดูลสกัดโครงสร้างข้อมูลทุนวิจัย (Typed-Claim Extractor) ของระบบ COCO-RISE\n"
    "หน้าที่ของคุณคือคืนค่า JSON ตามโครงสร้าง TypedClaimBundle เท่านั้น ห้ามเขียนประโยคอธิบายอิสระ\n"
    "กติกาเหล็ก:\n"
    "1. คัดลอกค่าทุกช่องจาก verified_payload เท่านั้น ห้ามแก้ไขตัวเลข ชื่อทุน ชั้นทุน ปีงบประมาณ หรือหมายเหตุอัตรา\n"
    "2. ห้ามตัดแถวใน claims ออกแม้แต่แถวเดียว และห้ามเพิ่มคีย์อื่นนอกเหนือจากโครงสร้างที่กำหนด\n"
    "3. ตอบกลับเป็น JSON object เพียงอย่างเดียว"
)


def typed_claim_messages(slots: dict, used: list[dict], ruling: dict) -> list[dict]:
    """Build prompt messages for typed-claim selection without exposing raw user prose."""
    canonical = bundle_from_verified(slots, used, ruling)
    payload = canonical.model_dump(
        mode="json",
        exclude_none=True,
        exclude={"claims": {"__all__": {"unit"}}},
    )
    return [
        {"role": "system", "content": CLAIM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "คัดลอกและยืนยันโครงสร้าง TypedClaimBundle จากข้อมูลที่ตรวจแล้วต่อไปนี้เป็น JSON เท่านั้น:\n"
                f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
            ),
        },
    ]


def may_emit_typed_claims(
    used: list[dict],
    evidence: dict,
    slots: dict | None = None,
    answer_mode: str = "verified_template",
    model: str | None = None,
) -> bool:
    """Return True only when verified structured facts may pass through typed-claim selection."""
    if answer_mode != "typed_claim" or model == "template":
        return False
    if not llm.enabled():
        return False
    if not used or any(not store.is_verified(f) or not f.get("fund") or not f.get("amount_type") for f in used):
        return False
    if not slots:
        return False
    if (evidence.get("pending_review") and slots.get("intent") == "amount") or evidence.get("ceiling_fallback"):
        return False
    if slots.get("intent") not in ("amount", "ceiling", "eligibility"):
        return False
    if slots.get("alternatives") or slots.get("clause_conflict"):
        return False
    return True


def may_emit_passage_claim(
    used: list[dict],
    citations: list[str],
    evidence: dict,
    slots: dict | None = None,
    answer_mode: str = "verified_template",
    model: str | None = None,
) -> bool:
    """Return True when a verified non-fact answer (definition, details, tiers, list, eligibility) may use TypedPassageClaim."""
    if answer_mode not in ("typed_claim", "llm") or model == "template":
        return False
    if not llm.enabled():
        return False
    if not citations or any(not store.is_verified(f) for f in used):
        return False
    if not slots:
        return False
    if (evidence.get("pending_review") and slots.get("intent") == "amount") or evidence.get("ceiling_fallback"):
        return False
    intent = slots.get("intent")
    if intent == "eligibility" and used:
        return False
    if intent not in ("definition", "details", "tiers", "list", "catalogue_details", "eligibility"):
        return False
    if slots.get("alternatives") or slots.get("clause_conflict"):
        return False
    return True


def validate_typed_claims(
    raw_text: str,
    slots: dict,
    used: list[dict],
    ruling: dict,
    evidence: dict,
) -> tuple[bool, list[str], TypedClaimBundle | None]:
    """Validate an LLM typed-claim response 1:1 against verified evidence and ruling."""
    problems: list[str] = []
    if not used:
        return False, ["ไม่มีข้อเท็จจริงที่ตรวจแล้วสำหรับสร้าง typed claims"], None
    if any(not store.is_verified(fact) for fact in used):
        return False, ["ข้อมูลต้นทางยังไม่ผ่านการอนุมัติ"], None
    if evidence.get("pending_review") and slots.get("intent") == "amount":
        return False, ["ข้อมูลปีงบประมาณยังรอการอนุมัติ จึงไม่ใช้โมเดล"], None
    if evidence.get("ceiling_fallback"):
        return False, ["คำตอบแบบ fallback เพดานกลางต้องใช้ข้อความมาตรฐานเท่านั้น"], None

    cleaned = _clean_json_text(raw_text)
    if not cleaned:
        return False, ["โมเดลไม่ได้คืนค่า JSON"], None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return False, ["คำตอบจากโมเดลไม่ใช่ JSON ที่ถูกต้อง"], None
    if not isinstance(parsed, dict):
        return False, ["คำตอบจากโมเดลต้องเป็น JSON object"], None
    try:
        bundle = TypedClaimBundle.model_validate(parsed)
    except ValidationError as exc:
        fields = ", ".join(".".join(str(part) for part in err["loc"]) for err in exc.errors())
        return False, [f"โครงสร้าง TypedClaimBundle ไม่ถูกต้อง ({fields or 'schema mismatch'})"], None

    expected_intent = slots.get("intent")
    if expected_intent in ("amount", "ceiling", "eligibility", "tiers", "details") and bundle.intent != expected_intent:
        problems.append(f"intent ใน claim ({bundle.intent}) ไม่ตรงกับคำถาม ({expected_intent})")
    if bundle.year_assumed != bool(slots.get("year_assumed", False)):
        problems.append("สถานะ year_assumed ไม่ตรงกับคำถาม")
    if len(bundle.claims) != len(used):
        problems.append(f"จำนวนแถวใน claims ({len(bundle.claims)}) ไม่ตรงกับหลักฐาน ({len(used)})")
        return False, problems, None

    for idx, (claim, fact) in enumerate(zip(bundle.claims, used), start=1):
        source = fact.get("source") or {}
        if claim.fact_id is not None and fact.get("id") is not None and claim.fact_id != fact.get("id"):
            problems.append(f"แถวที่ {idx}: fact_id ไม่ตรงกับหลักฐาน")
        if claim.fund != fact.get("fund"):
            problems.append(f"แถวที่ {idx}: ชื่อทุน {claim.fund} ไม่ตรงกับหลักฐาน {fact.get('fund')}")
        if claim.tier != fact.get("tier"):
            problems.append(f"แถวที่ {idx}: ชั้นทุน {claim.tier} ไม่ตรงกับหลักฐาน {fact.get('tier')}")
        if claim.amount_type != fact.get("amount_type"):
            problems.append(
                f"แถวที่ {idx}: ชนิดจำนวนเงิน {claim.amount_type} ไม่ตรงกับหลักฐาน {fact.get('amount_type')}"
            )
        if claim.fiscal_year != fact.get("fiscal_year"):
            problems.append(
                f"แถวที่ {idx}: ปีงบประมาณ {claim.fiscal_year} ไม่ตรงกับหลักฐาน {fact.get('fiscal_year')}"
            )
        if (
            fact.get("amount_type") == "allocated"
            and slots.get("fiscal_year")
            and claim.fiscal_year != slots.get("fiscal_year")
        ):
            problems.append(f"แถวที่ {idx}: ปีงบประมาณของวงเงินจัดสรรไม่ตรงกับปีที่ถาม")
        if claim.amount != fact.get("amount"):
            problems.append(f"แถวที่ {idx}: จำนวนเงิน {claim.amount} ไม่ตรงกับหลักฐาน {fact.get('amount')}")
        if claim.total_cap != fact.get("total_cap"):
            problems.append(f"แถวที่ {idx}: วงเงินรวม {claim.total_cap} ไม่ตรงกับหลักฐาน {fact.get('total_cap')}")
        if claim.max_grants_per_person != fact.get("max_grants_per_person"):
            problems.append(f"แถวที่ {idx}: จำนวนทุนต่อคนไม่ตรงกับหลักฐาน")
        if claim.max_grants_per_year != fact.get("max_grants_per_year"):
            problems.append(f"แถวที่ {idx}: จำนวนทุนต่อปีไม่ตรงกับหลักฐาน")
        if (claim.note or None) != (fact.get("note") or None):
            problems.append(f"แถวที่ {idx}: หมายเหตุอัตรา (note) ไม่ตรงกับหลักฐาน")
        if claim.source_doc != source.get("doc"):
            problems.append(f"แถวที่ {idx}: เอกสารอ้างอิง {claim.source_doc} ไม่ตรงกับหลักฐาน {source.get('doc')}")
        if claim.source_clause != source.get("clause"):
            problems.append(f"แถวที่ {idx}: ข้ออ้างอิง {claim.source_clause} ไม่ตรงกับหลักฐาน {source.get('clause')}")
        if claim.source_page != source.get("page"):
            problems.append(f"แถวที่ {idx}: หน้าอ้างอิง {claim.source_page} ไม่ตรงกับหลักฐาน {source.get('page')}")

    expected_verdict = (ruling or {}).get("verdict") or "no_rule"
    if bundle.verdict != expected_verdict:
        problems.append(f"ผลการตรวจสิทธิ์ {bundle.verdict} ไม่ตรงกับกฎ ({expected_verdict})")
    if (bundle.verdict_reason or None) != ((ruling or {}).get("reason") or None):
        problems.append("เหตุผลการตรวจสิทธิ์ไม่ตรงกับกฎที่ตรวจแล้ว")
    ruling_source = (ruling or {}).get("source") or {}
    if ruling_source:
        if bundle.verdict_source_doc != ruling_source.get("doc"):
            problems.append("เอกสารอ้างอิงของกฎไม่ตรงกับหลักฐาน")
        if bundle.verdict_source_clause != ruling_source.get("clause"):
            problems.append("ข้ออ้างอิงของกฎไม่ตรงกับหลักฐาน")
        if bundle.verdict_source_page != ruling_source.get("page"):
            problems.append("หน้าอ้างอิงของกฎไม่ตรงกับหลักฐาน")

    if problems:
        return False, problems, None
    return True, [], bundle


def render_typed_claims(
    bundle: TypedClaimBundle,
    slots: dict,
    evidence: dict,
    ruling: dict,
    profile: dict | None = None,
) -> tuple[str, list[str], list[dict]]:
    """Deterministically render a validated TypedClaimBundle into Thai prose and citations."""
    reconstructed_facts: list[dict] = []
    for claim in bundle.claims:
        reconstructed_facts.append(
            {
                "id": claim.fact_id,
                "fund": claim.fund,
                "tier": claim.tier,
                "fiscal_year": claim.fiscal_year,
                "amount_type": claim.amount_type,
                "amount": claim.amount,
                "total_cap": claim.total_cap,
                "max_grants_per_person": claim.max_grants_per_person,
                "max_grants_per_year": claim.max_grants_per_year,
                "note": claim.note,
                "verified": True,
                "source": {
                    "doc": claim.source_doc,
                    "clause": claim.source_clause,
                    "page": claim.source_page,
                },
            }
        )
    reconstructed_ruling = {
        "verdict": bundle.verdict,
        "reason": bundle.verdict_reason,
        "source": (
            {
                "doc": bundle.verdict_source_doc,
                "clause": bundle.verdict_source_clause,
                "page": bundle.verdict_source_page,
            }
            if bundle.verdict_source_doc and bundle.verdict_source_clause
            else (ruling or {}).get("source")
        ),
        "missing": list((ruling or {}).get("missing") or []),
    }
    claim_evidence = {
        **evidence,
        "facts": reconstructed_facts,
        "ceiling_fallback": False,
        "pending_review": False,
    }
    claim_slots = {**slots, "year_assumed": bundle.year_assumed}
    rendered, citations, used = compose_answer(
        claim_slots, claim_evidence, reconstructed_ruling, profile
    )
    before, after = answers.name_notes(claim_slots)
    full_answer = " ".join(part for part in (before, rendered, after) if part)
    return full_answer, citations, used


def request_typed_claims(
    slots: dict,
    used: list[dict],
    ruling: dict,
    evidence: dict,
    model: str | None = None,
    timeout: float = 0.0,
) -> tuple[bool, list[str], TypedClaimBundle | None, str]:
    """Call the local LLM in JSON mode to produce a TypedClaimBundle and validate it."""
    global _CLAIM_FAILURE_UNTIL
    chosen_model = model or llm.SMALL or llm.MAIN
    with _CLAIM_CIRCUIT_LOCK:
        if time.monotonic() < _CLAIM_FAILURE_UNTIL:
            return False, ["วงจรเรียกโมเดลพักชั่วคราวหลังเกิดข้อผิดพลาด"], None, chosen_model
    messages = typed_claim_messages(slots, used, ruling)
    try:
        raw = llm.chat(
            messages,
            model=chosen_model,
            max_tokens=320,
            json_mode=True,
            timeout=timeout or CLAIM_TIMEOUT,
        )
        with _CLAIM_CIRCUIT_LOCK:
            _CLAIM_FAILURE_UNTIL = 0.0
    except Exception as exc:
        with _CLAIM_CIRCUIT_LOCK:
            _CLAIM_FAILURE_UNTIL = time.monotonic() + CLAIM_RETRY_SECONDS
        return False, [f"เรียกโมเดลไม่สำเร็จ: {type(exc).__name__}"], None, chosen_model

    ok, problems, bundle = validate_typed_claims(raw, slots, used, ruling, evidence)
    return ok, problems, bundle, chosen_model


def passage_claim_messages(slots: dict, citations: list[str]) -> list[dict]:
    """Build compact JSON-mode messages for verified passage, definition, or catalogue claims."""
    payload = {
        "intent": str(slots.get("intent") or "details"),
        "fund": slots.get("fund"),
        "citation_count": len(citations),
        "primary_citation": citations[0] if citations else "",
    }
    return [
        {
            "role": "system",
            "content": (
                "คุณคือโมดูลยืนยันโครงสร้างหลักฐาน (Typed-Passage Claim Extractor) ของระบบ COCO-RISE\n"
                "คืนค่า JSON ตามโครงสร้าง TypedPassageClaim จากข้อมูลที่ตรวจแล้วเท่านั้น ห้ามเขียนข้อความอิสระ"
            ),
        },
        {
            "role": "user",
            "content": (
                "ยืนยันโครงสร้าง TypedPassageClaim ต่อไปนี้เป็น JSON เท่านั้น:\n"
                f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
            ),
        },
    ]


def validate_passage_claim(
    raw_text: str,
    slots: dict,
    citations: list[str],
) -> tuple[bool, list[str], TypedPassageClaim | None]:
    """Validate a TypedPassageClaim JSON against the verified slot and citation metadata."""
    if not citations:
        return False, ["ไม่มีแหล่งอ้างอิงที่ตรวจแล้ว"], None
    cleaned = _clean_json_text(raw_text)
    if not cleaned:
        return False, ["โมเดลไม่ได้คืนค่า JSON"], None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return False, ["คำตอบจากโมเดลไม่ใช่ JSON ที่ถูกต้อง"], None
    if not isinstance(parsed, dict):
        return False, ["คำตอบจากโมเดลต้องเป็น JSON object"], None
    try:
        claim = TypedPassageClaim.model_validate(parsed)
    except ValidationError as exc:
        fields = ", ".join(".".join(str(part) for part in err["loc"]) for err in exc.errors())
        return False, [f"โครงสร้าง TypedPassageClaim ไม่ถูกต้อง ({fields or 'schema mismatch'})"], None

    problems: list[str] = []
    if claim.intent != str(slots.get("intent") or ""):
        problems.append("intent ใน passage claim ไม่ตรงกับคำถาม")
    if (claim.fund or None) != (slots.get("fund") or None):
        problems.append("fund ใน passage claim ไม่ตรงกับคำถาม")
    if claim.citation_count != len(citations):
        problems.append("จำนวนแหล่งอ้างอิงใน passage claim ไม่ตรงกับหลักฐาน")
    if claim.primary_citation != citations[0]:
        problems.append("แหล่งอ้างอิงหลักใน passage claim ไม่ตรงกับหลักฐาน")
    if problems:
        return False, problems, None
    return True, [], claim


def request_passage_claim(
    slots: dict,
    citations: list[str],
    model: str | None = None,
    timeout: float = 0.0,
) -> tuple[bool, list[str], TypedPassageClaim | None, str]:
    """Call the local LLM in JSON mode to validate a TypedPassageClaim."""
    global _CLAIM_FAILURE_UNTIL
    chosen_model = model or llm.SMALL or llm.MAIN
    with _CLAIM_CIRCUIT_LOCK:
        if time.monotonic() < _CLAIM_FAILURE_UNTIL:
            return False, ["วงจรเรียกโมเดลพักชั่วคราวหลังเกิดข้อผิดพลาด"], None, chosen_model
    messages = passage_claim_messages(slots, citations)
    try:
        raw = llm.chat(
            messages,
            model=chosen_model,
            max_tokens=120,
            json_mode=True,
            timeout=timeout or CLAIM_TIMEOUT,
        )
        with _CLAIM_CIRCUIT_LOCK:
            _CLAIM_FAILURE_UNTIL = 0.0
    except Exception as exc:
        with _CLAIM_CIRCUIT_LOCK:
            _CLAIM_FAILURE_UNTIL = time.monotonic() + CLAIM_RETRY_SECONDS
        return False, [f"เรียกโมเดลไม่สำเร็จ: {type(exc).__name__}"], None, chosen_model

    ok, problems, claim = validate_passage_claim(raw, slots, citations)
    return ok, problems, claim, chosen_model


def reset_circuit() -> None:
    """Reset the typed-claim circuit breaker (used in tests)."""
    global _CLAIM_FAILURE_UNTIL
    with _CLAIM_CIRCUIT_LOCK:
        _CLAIM_FAILURE_UNTIL = 0.0


# ---------------------------------------------------------------- Step 1: closed-schema slot extraction
SLOT_SYSTEM_PROMPT = (
    "คุณคือโมดูลสกัดสล็อตคำถาม (Closed-Schema Slot Extractor) สำหรับระบบถามตอบทุนวิจัย\n"
    "ตอบเป็น JSON ตามโครงสร้าง {\"fund\": ..., \"tier\": ..., \"fiscal_year\": ..., \"intent\": ...} เท่านั้น\n"
    "กติกา:\n"
    "- fund ต้องเป็นชื่อทุนที่ปรากฏในรายการทุนที่กำหนด หรือ null ถ้าคำถามไม่ได้ระบุทุนชัดเจน\n"
    "- tier ต้องเป็น ชั้นต้น, ชั้นกลาง, ชั้นสูง หรือ null เท่านั้น\n"
    "- fiscal_year ต้องเป็นเลขปี พ.ศ. 4 หลักที่ปรากฏในคำถาม หรือ null\n"
    "- intent ต้องเป็นหนึ่งใน: amount, ceiling, tiers, details, definition, eligibility, application, list, catalogue_details หรือ null /no_think"
)


def slot_extraction_messages(question: str, known_fund: str | None = None) -> list[dict]:
    safe_question = limits.redact_pii(question)
    if known_fund:
        candidates = [known_fund]
    else:
        candidates = [f for f in store.view().FUNDS if _fund_grounded_in_question(f, safe_question)]
    funds_hint = f"ทุนที่เกี่ยวข้อง: {', '.join(candidates)}\n" if candidates else ""
    return [
        {"role": "system", "content": SLOT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{funds_hint}"
                f"คำถาม: {safe_question}\n"
                "หากผู้ถามระบุแต่ชื่อทุนโดยยังไม่ระบุคำกริยาถามชัดเจน ให้เลือก intent เป็น details\n"
                "คืนค่า JSON สล็อตของคำถามนี้เท่านั้น"
            ),
        },
    ]


def _fund_grounded_in_question(fund: str, question: str) -> bool:
    """Ensure an LLM-extracted fund has a real lexical anchor in the user's question."""
    if fund in question:
        return True
    core = fund.removeprefix("ทุน").removeprefix("เงินรางวัล").strip()
    if len(core) >= 4 and core in question:
        return True
    for alias, target in getattr(store.view(), "ALIASES", ()):
        if target == fund and len(alias) >= 4 and alias in question:
            return True
    tokens = [tok for tok in re.split(r"\s+", core) if len(tok) >= 4]
    return any(tok in question for tok in tokens)


def validate_slot_claim(raw_text: str, question: str) -> TypedSlotClaim | None:
    """Validate an LLM slot extraction JSON against closed vocabularies and question anchors."""
    cleaned = _clean_json_text(raw_text)
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("tier") in store.TIER_ALIAS:
        parsed = {**parsed, "tier": store.TIER_ALIAS[parsed["tier"]]}
    try:
        claim = TypedSlotClaim.model_validate(parsed)
    except ValidationError:
        return None

    if claim.fund is not None:
        if claim.fund not in store.view().FUNDS or not _fund_grounded_in_question(claim.fund, question):
            return None
    if claim.tier is not None:
        canonical_tiers = set(store.TIERS) | set(store.TIER_ALIAS.values())
        if claim.tier not in canonical_tiers or not any(
            t in question for t in (*store.TIERS, *store.TIER_ALIAS.keys())
        ):
            claim.tier = None
    if claim.fiscal_year is not None:
        written_years = {int(m) for m in YEAR_TOKEN_RE.findall(question)}
        if not (2500 <= claim.fiscal_year <= 2700) or claim.fiscal_year not in written_years:
            claim.fiscal_year = None
    if claim.intent is not None and claim.intent not in ALLOWED_SLOT_INTENTS:
        claim.intent = None
    if claim.fund is None and claim.tier is None and claim.fiscal_year is None and claim.intent is None:
        return None
    return claim


def extract_slots_with_llm(
    question: str,
    model: str | None = None,
    timeout: float = 0.0,
    known_fund: str | None = None,
) -> TypedSlotClaim | None:
    """Optional Step 1 fallback: extract validated closed-schema slots when rules leave intent unknown."""
    if not llm.enabled():
        return None
    chosen_model = model or llm.SMALL or llm.MAIN
    try:
        raw = llm.chat(
            slot_extraction_messages(question, known_fund=known_fund),
            model=chosen_model,
            max_tokens=48,
            json_mode=True,
            timeout=timeout or SLOT_TIMEOUT,
        )
    except Exception:
        return None
    return validate_slot_claim(raw, question)


# ---------------------------------------------------------------- Step 3: RAG LLM Answer Generator (verified by Step 4)
RAG_GENERATOR_SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยตอบคำถามระเบียบทุนวิจัยของวิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น\n"
    "เขียนวลีเปิดนำคำตอบสั้นๆ 1 วลี (5-10 คำ) ที่สุภาพ เป็นทางการ และตรงหัวข้อหลักฐาน\n"
    "กติกา: ห้ามตั้งคำถามกลับ (ห้ามมี ? ไหม หรือไม่ อย่างไร) ห้ามใส่ตัวเลข ห้ามนิยามความหมายขึ้นมาเอง\n"
    "ตัวอย่าง: {\"lead\": \"จากข้อมูลในประกาศของวิทยาลัย สรุปได้ดังนี้ครับ\"} /no_think"
)


def may_generate_llm_answer(
    used: list[dict],
    citations: list[str],
    evidence: dict,
    slots: dict | None = None,
    answer_mode: str = "verified_template",
    model: str | None = None,
) -> bool:
    """Return True when the LLM may synthesize natural Thai prose from verified RAG evidence."""
    if answer_mode != "llm" or model == "template":
        return False
    if not llm.enabled():
        return False
    if not citations or any(not store.is_verified(f) for f in used):
        return False
    if not slots:
        return False
    intent = slots.get("intent")
    if intent in (
        "clarify_reference",
        "smalltalk",
        "history",
    ):
        return False
    if slots.get("alternatives") or slots.get("clause_conflict"):
        return False
    return True


def rag_generation_messages(
    question: str,
    slots: dict,
    used: list[dict],
    ruling: dict,
    evidence: dict,
    reference_answer: str,
) -> list[dict]:
    """Build compact RAG prompt messages for natural Thai answer synthesis (zero PII sent to LLM)."""
    fund_hint = slots.get("fund") or "ข้อมูลตามประกาศ"
    safe_question = limits.redact_pii(question)
    user_prompt = (
        f"คำถาม: {safe_question}\n"
        f"หัวข้อหลักฐาน: {fund_hint}\n"
        "คืนค่า JSON {\"lead\": \"...\"} เป็นวลีเปิดนำคำตอบที่เป็นทางการ (ห้ามตั้งคำถาม ห้ามใส่ตัวเลข):"
    )
    return [
        {"role": "system", "content": RAG_GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def validate_llm_prose(
    text: str,
    slots: dict,
    used: list[dict],
    reference_answer: str,
) -> list[str]:
    """Additional semantic/grounding checks for LLM-generated prose when used_facts is empty or partial."""
    problems: list[str] = []
    cleaned = text.strip()
    if not cleaned or len(cleaned) < 15:
        problems.append("คำตอบจากโมเดลสั้นผิดปกติหรือว่างเปล่า")
        return problems
    fund = slots.get("fund")
    if fund and fund not in cleaned:
        problems.append(f"คำตอบจากโมเดลไม่ได้ระบุชื่อทุน {fund}")
    if slots.get("intent") == "details" and "รอเจ้าหน้าที่ตรวจ" in reference_answer:
        if "รอ" not in cleaned or "อนุมัติ" not in cleaned:
            problems.append("คำตอบจากโมเดลตกหล่นสถานะเงื่อนไขที่ยังรอเจ้าหน้าที่ตรวจและอนุมัติ")
    if not used and slots.get("intent") not in ("list", "catalogue_details", "open_now", "recommendation"):
        allowed_funds = {fund} if fund else set()
        strays = [
            f for f in store.view().FUNDS
            if f not in allowed_funds and f not in reference_answer and f in cleaned
        ]
        if strays:
            problems.append(f"คำตอบจากโมเดลกล่าวถึงทุนอื่นที่ไม่อยู่ในคำถาม ({strays[0]})")
    return problems


def request_llm_rag_answer(
    question: str,
    slots: dict,
    used: list[dict],
    ruling: dict,
    evidence: dict,
    reference_answer: str,
    model: str | None = None,
    timeout: float = 0.0,
) -> tuple[bool, list[str], str | None, str]:
    """Call the local LLM to write natural Thai prose from verified RAG evidence."""
    global _CLAIM_FAILURE_UNTIL
    chosen_model = model or llm.SMALL or llm.MAIN
    with _CLAIM_CIRCUIT_LOCK:
        if time.monotonic() < _CLAIM_FAILURE_UNTIL:
            return False, ["วงจรเรียกโมเดลพักชั่วคราวหลังเกิดข้อผิดพลาด"], None, chosen_model
    messages = rag_generation_messages(question, slots, used, ruling, evidence, reference_answer)
    try:
        raw = llm.chat(
            messages,
            model=chosen_model,
            max_tokens=64,
            json_mode=True,
            timeout=timeout or CLAIM_TIMEOUT,
        )
        with _CLAIM_CIRCUIT_LOCK:
            _CLAIM_FAILURE_UNTIL = 0.0
    except Exception as exc:
        with _CLAIM_CIRCUIT_LOCK:
            _CLAIM_FAILURE_UNTIL = time.monotonic() + CLAIM_RETRY_SECONDS
        return False, [f"เรียกโมเดลไม่สำเร็จ: {type(exc).__name__}"], None, chosen_model

    cleaned_json = _clean_json_text(raw)
    answer_text = ""
    if cleaned_json:
        try:
            parsed = json.loads(cleaned_json)
            if isinstance(parsed, dict):
                for key in ("answer", "response", "text", "คำตอบ", "evidence"):
                    val = parsed.get(key)
                    if isinstance(val, str) and len(val.strip()) >= 15:
                        answer_text = val.strip()
                        break
                if not answer_text:
                    lead = parsed.get("lead")
                    if not isinstance(lead, str) or not lead.strip():
                        lead = next((v for v in parsed.values() if isinstance(v, str) and v.strip()), "")
                    if isinstance(lead, str) and lead.strip():
                        lead_clean = re.sub(r"\d+", "", lead).strip().rstrip(":：")
                        allowed_fund = slots.get("fund")
                        bad_tokens = ("?", "？", "ไหม", "หรือไม่", "อย่างไร", "เพื่อน", "คือการ", "หมายถึงการ")
                        if any(tok in lead_clean for tok in bad_tokens) or any(
                            f != allowed_fund and f in lead_clean for f in store.view().FUNDS
                        ):
                            lead_clean = (
                                f"สำหรับ{allowed_fund} สรุปข้อมูลตามประกาศได้ดังนี้ครับ"
                                if allowed_fund
                                else "จากข้อมูลในประกาศ สรุปได้ดังนี้ครับ"
                            )
                        if lead_clean:
                            answer_text = f"{lead_clean} {reference_answer}"
        except Exception:
            m = re.search(r'"lead"\s*:\s*"([^"\r\n]+)', cleaned_json)
            if m:
                lead_clean = re.sub(r"\d+", "", m.group(1)).strip().rstrip(":：")
                if lead_clean:
                    answer_text = f"{lead_clean} {reference_answer}"
            else:
                answer_text = ""
    if not answer_text:
        return False, ["โมเดลไม่ได้คืนข้อความคำตอบใน JSON {\"answer\": \"...\"}"], None, chosen_model
    problems = validate_llm_prose(answer_text, slots, used, reference_answer)
    if problems:
        return False, problems, None, chosen_model
    return True, [], answer_text, chosen_model
