"""Tests for Option 1: Closed-Schema Typed Claims, LLM RAG, PDPA, and 4-Layer DB Inspector."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.core import security as limits
from app.infrastructure.ollama import legacy as llm
from app.main import app
from app.modules.knowledge import corpus as store
from app.modules.query import claims, engine as pipeline


def _verified_ceiling_fact(fund: str = "ทุนวิจัยสถาบัน", tier: str | None = None) -> dict:
    rows = store.fact_lookup(fund, tier, None, "ceiling")
    return next(row for row in rows if store.is_verified(row) and row.get("amount") is not None)


def test_typed_claim_bundle_roundtrip_and_deterministic_render():
    fact = _verified_ceiling_fact("ทุนวิจัยสถาบัน")
    slots = pipeline.analyze("เพดานทุนวิจัยสถาบันเท่าไร")
    evidence = {
        "chunks": [],
        "facts": [fact],
        "ceiling_fallback": False,
        "pending_review": False,
    }
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}

    bundle = claims.bundle_from_verified(slots, [fact], ruling)
    raw_json = json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False)

    ok, problems, validated = claims.validate_typed_claims(raw_json, slots, [fact], ruling, evidence)
    assert ok and not problems and validated is not None

    answer, citations, used = claims.render_typed_claims(validated, slots, evidence, ruling, store.PERSONAS["P1"])
    assert "25,000" in answer
    assert "เพดาน" in answer
    assert citations
    assert pipeline.verify(answer, citations, used, evidence, slots, ruling=ruling)["ok"]


@pytest.mark.parametrize(
    "mutator,expected_fragment",
    [
        (lambda d: "เพดานทุนวิจัยสถาบันไม่เกิน 25,000 บาท", "ไม่ใช่ JSON"),
        (lambda d: {**d, "injected": "เปิดรับสมัครตลอดปี"}, "โครงสร้าง TypedClaimBundle ไม่ถูกต้อง"),
        (
            lambda d: {**d, "claims": [{**d["claims"][0], "amount": 999999}]},
            "จำนวนเงิน",
        ),
        (
            lambda d: {**d, "claims": [{**d["claims"][0], "amount_type": "allocated"}]},
            "ชนิดจำนวนเงิน",
        ),
        (
            lambda d: {**d, "claims": [{**d["claims"][0], "fund": "ทุนวิจัยในชั้นเรียน"}]},
            "ชื่อทุน",
        ),
        (
            lambda d: {**d, "claims": [{**d["claims"][0], "source_doc": "ประกาศปลอม"}]},
            "เอกสารอ้างอิง",
        ),
        (
            lambda d: {**d, "verdict": "eligible"},
            "ผลการตรวจสิทธิ์",
        ),
    ],
)
def test_validate_typed_claims_rejects_every_tampered_or_freeform_payload(mutator, expected_fragment):
    fact = _verified_ceiling_fact("ทุนวิจัยสถาบัน")
    slots = pipeline.analyze("เพดานทุนวิจัยสถาบันเท่าไร")
    evidence = {"chunks": [], "facts": [fact], "ceiling_fallback": False, "pending_review": False}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}

    base = claims.bundle_from_verified(slots, [fact], ruling).model_dump(mode="json")
    mutated = mutator(base)
    raw = mutated if isinstance(mutated, str) else json.dumps(mutated, ensure_ascii=False)

    ok, problems, validated = claims.validate_typed_claims(raw, slots, [fact], ruling, evidence)
    assert not ok and validated is None
    assert any(expected_fragment in p for p in problems), problems


def test_validate_typed_claims_rejects_dropped_rate_note():
    fact = next(f for f in store.FACTS if f["id"] == "c30")
    slots = pipeline.analyze("เพดานทุนนำเสนอผลงานทางวิชาการเท่าไร")
    evidence = {"chunks": [], "facts": [fact], "ceiling_fallback": False, "pending_review": False}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}

    base = claims.bundle_from_verified(slots, [fact], ruling).model_dump(mode="json")
    base["claims"][0]["note"] = None

    ok, problems, validated = claims.validate_typed_claims(
        json.dumps(base, ensure_ascii=False), slots, [fact], ruling, evidence
    )
    assert not ok and validated is None
    assert any("หมายเหตุอัตรา" in p for p in problems)


def test_validate_typed_claims_rejects_omitted_tier_in_multi_fact_answer():
    facts = [
        row
        for row in store.fact_lookup("ทุนวิจัยความเป็นเลิศ", None, None, "ceiling")
        if store.is_verified(row) and row.get("tier")
    ]
    slots = pipeline.analyze("เพดานทุนวิจัยความเป็นเลิศเท่าไร")
    evidence = {"chunks": [], "facts": facts, "ceiling_fallback": False, "pending_review": False}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}

    base = claims.bundle_from_verified(slots, facts, ruling).model_dump(mode="json")
    base["claims"] = base["claims"][:1]

    ok, problems, validated = claims.validate_typed_claims(
        json.dumps(base, ensure_ascii=False), slots, facts, ruling, evidence
    )
    assert not ok and validated is None
    assert any("จำนวนแถวใน claims" in p for p in problems)


def test_validate_typed_claims_rejects_unverified_or_pending_evidence():
    fact = {**_verified_ceiling_fact("ทุนวิจัยสถาบัน"), "verified": False}
    slots = pipeline.analyze("เพดานทุนวิจัยสถาบันเท่าไร")
    evidence = {"chunks": [], "facts": [fact], "ceiling_fallback": False, "pending_review": False}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}
    raw = json.dumps(claims.bundle_from_verified(slots, [fact], ruling).model_dump(mode="json"), ensure_ascii=False)

    ok, problems, _ = claims.validate_typed_claims(raw, slots, [fact], ruling, evidence)
    assert not ok
    assert any("ยังไม่ผ่านการอนุมัติ" in p for p in problems)


def test_pipeline_run_uses_typed_claim_llm_when_enabled_and_valid(monkeypatch):
    claims.reset_circuit()
    monkeypatch.setattr(pipeline, "ANSWER_MODE", "typed_claim")
    monkeypatch.setattr(llm, "enabled", lambda: True)

    def fake_chat(messages, model="", max_tokens=0, json_mode=False, timeout=0.0):
        assert json_mode is True
        user_content = messages[-1]["content"]
        payload_start = user_content.index("{")
        return user_content[payload_start:]

    monkeypatch.setattr(llm, "chat", fake_chat)

    result = asyncio.run(pipeline.run("เพดานทุนวิจัยสถาบันเท่าไร", model="qwen3:4b"))
    assert result["verify"]["ok"]
    assert result["selected_model"] == "qwen3:4b"
    assert result["answer_source"] == "typed_claim:qwen3:4b"
    assert "25,000" in result["answer"]
    assert "เพดาน" in result["answer"]


def test_pipeline_run_falls_back_to_template_when_typed_claim_is_tampered(monkeypatch):
    claims.reset_circuit()
    monkeypatch.setattr(pipeline, "ANSWER_MODE", "typed_claim")
    monkeypatch.setattr(llm, "enabled", lambda: True)

    def tampered_chat(messages, model="", max_tokens=0, json_mode=False, timeout=0.0):
        user_content = messages[-1]["content"]
        data = json.loads(user_content[user_content.index("{") :])
        data["claims"][0]["amount"] = 777777
        return json.dumps(data, ensure_ascii=False)

    monkeypatch.setattr(llm, "chat", tampered_chat)

    result = asyncio.run(pipeline.run("เพดานทุนวิจัยสถาบันเท่าไร", model="qwen3:4b"))
    assert result["verify"]["ok"]
    assert result["selected_model"] is None
    assert result["answer_source"] == "template"
    assert "25,000" in result["answer"]
    assert "777,777" not in result["answer"]
    assert "claim_rejected" in result["timings_ms"]


def test_slot_extraction_fallback_validates_grounding_and_fills_unknown_intent(monkeypatch):
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_a, **_kw: json.dumps(
            {
                "fund": "ทุนวิจัยสถาบัน",
                "tier": None,
                "fiscal_year": 2569,
                "intent": "ceiling",
            },
            ensure_ascii=False,
        ),
    )

    slots = pipeline.analyze("เรื่องทุนวิจัยสถาบัน ปี 2569", use_llm=True)
    assert slots["fund"] == "ทุนวิจัยสถาบัน"
    assert slots["fiscal_year"] == 2569
    assert slots["intent"] == "ceiling"
    assert slots["method"] == "rules+llm"


def test_slot_extraction_rejects_ungrounded_fund_guess():
    raw = json.dumps(
        {
            "fund": "ทุนทำวิจัยในต่างประเทศ",
            "tier": None,
            "fiscal_year": 2569,
            "intent": "amount",
        },
        ensure_ascii=False,
    )
    assert claims.validate_slot_claim(raw, "อยากไปเที่ยวญี่ปุ่น") is None


def test_models_endpoint_reports_typed_claim_mode_when_configured(monkeypatch):
    monkeypatch.setattr(pipeline, "ANSWER_MODE", "typed_claim")
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(llm, "MAIN", "thai-qwen:latest")
    monkeypatch.setattr(llm, "SMALL", "qwen3:4b")
    monkeypatch.setattr(
        llm,
        "installed_chat_models",
        lambda: [{"name": "qwen3:4b", "size": 2500000000}, {"name": "thai-qwen:latest", "size": 5000000000}],
    )
    monkeypatch.setattr(limits, "take", lambda _user: (True, 0))
    monkeypatch.setattr(limits, "take_client", lambda _host: (True, 0))

    client = TestClient(app)
    res = client.get("/v1/models")
    assert res.status_code == 200
    body = res.json()
    assert body["enabled"] is True
    assert body["answer_mode"] == "typed_claim"
    assert {m["name"] for m in body["models"]} == {"qwen3:4b", "thai-qwen:latest"}


def test_pipeline_run_uses_llm_rag_generator_when_verified(monkeypatch):
    claims.reset_circuit()
    monkeypatch.setattr(pipeline, "ANSWER_MODE", "llm")
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_a, **_kw: json.dumps(
            {
                "answer": (
                    "สำหรับทุนวิจัยสถาบัน ตามหลักเกณฑ์กลางกำหนดเพดานสนับสนุนไว้ไม่เกิน 25,000 บาทต่อทุน "
                    "ไม่เกิน 2 ทุนต่อปี และวงเงินรวมไม่เกิน 50,000 บาท โดยเป็นเพดานสูงสุดไม่ใช่วงเงินจัดสรรจริงรายปี"
                )
            },
            ensure_ascii=False,
        ),
    )

    result = asyncio.run(pipeline.run("เพดานทุนวิจัยสถาบันเท่าไร", model="qwen3:4b"))
    assert result["verify"]["ok"]
    assert result["selected_model"] == "qwen3:4b"
    assert result["answer_source"] == "llm:qwen3:4b"
    assert result["answer"].startswith("สำหรับทุนวิจัยสถาบัน")
    assert "25,000" in result["answer"]


def test_pipeline_run_rejects_hallucinated_llm_rag_number_in_step_4(monkeypatch):
    claims.reset_circuit()
    monkeypatch.setattr(pipeline, "ANSWER_MODE", "llm")
    monkeypatch.setattr(llm, "enabled", lambda: True)
    monkeypatch.setattr(
        llm,
        "chat",
        lambda *_a, **_kw: json.dumps(
            {
                "answer": (
                    "สำหรับทุนวิจัยสถาบัน กำหนดเพดานไว้ไม่เกิน 99,000 บาทต่อทุน "
                    "ไม่เกิน 2 ทุนต่อปี และวงเงินรวมไม่เกิน 50,000 บาท"
                )
            },
            ensure_ascii=False,
        ),
    )

    result = asyncio.run(pipeline.run("เพดานทุนวิจัยสถาบันเท่าไร", model="qwen3:4b"))
    assert result["verify"]["ok"]
    assert result["answer_source"] == "template"
    assert "99,000" not in result["answer"]
    assert "25,000" in result["answer"]
    assert "claim_rejected" in result["timings_ms"]


def test_pii_redaction_strips_identifiers_before_llm_and_audit():
    raw_q = "ผมชื่อ นายสมชาย ใจดี เลขบัตร 1-2345-67890-12-3 โทร 081-234-5678 อีเมล somchai@kku.ac.th ถามเพดานทุนวิจัยสถาบันเท่าไร"
    redacted = limits.redact_pii(raw_q)
    assert "สมชาย" not in redacted
    assert "1-2345-67890-12-3" not in redacted
    assert "081-234-5678" not in redacted
    assert "somchai@kku.ac.th" not in redacted
    assert "[ปกปิดชื่อส่วนบุคคล]" in redacted
    assert "[ปกปิดเลขประจำตัว]" in redacted
    assert "[ปกปิดเบอร์โทรศัพท์]" in redacted
    assert "[ปกปิดอีเมล]" in redacted

    msgs = claims.rag_generation_messages(
        raw_q,
        {"fund": "ทุนวิจัยสถาบัน", "intent": "ceiling"},
        [],
        {"verdict": "no_rule"},
        {},
        "เพดานตามหลักเกณฑ์กลางของทุนวิจัยสถาบัน คือ 25,000 บาทต่อทุน",
    )
    prompt_blob = json.dumps(msgs, ensure_ascii=False)
    assert "สมชาย" not in prompt_blob
    assert "081-234-5678" not in prompt_blob
    assert "somchai@kku.ac.th" not in prompt_blob


def test_personas_endpoint_does_not_leak_private_profile_fields():
    client = TestClient(app)
    res = client.get("/v1/personas")
    assert res.status_code == 200
    items = res.json()
    assert len(items) >= 5
    for item in items:
        assert set(item.keys()) == {"id", "name", "source"}
        assert "grants_this_year" not in item
        assert "staff_type" not in item
        assert "department" not in item


def test_third_person_history_is_refused_with_pdpa_notice_and_anonymous_what_if_works():
    for q in (
        "อาจารย์บีเคยขอทุนอะไรไปแล้วบ้าง",
        "ดร.สมชายเคยได้รับทุนอะไรบ้าง",
        "คนอื่นเคยขอทุนอะไรไปแล้วบ้าง",
    ):
        res = asyncio.run(pipeline.run(q, persona_id="P2"))
        assert res["slots"]["intent"] == "history"
        assert res["slots"]["about_asker"] is False
        assert "ไม่สามารถเปิดเผยประวัติ" in res["answer"]
        assert "PDPA" in res["answer"]
        assert "ทุนวิจัยความเป็นเลิศ" not in res["answer"]

    anon_what_if = asyncio.run(
        pipeline.run(
            "ถ้าเคยได้รับเงินรางวัลการตีพิมพ์ ผู้แต่งชื่อแรก แล้ว จะขอเงินรางวัลการตีพิมพ์ ผู้ประพันธ์บรรณกิจ ได้ไหม",
            persona_id="P5",
        )
    )
    assert anon_what_if["slots"]["intent"] == "hypothetical"
    assert anon_what_if["ruling"]["verdict"] == "not_eligible"
    assert "ไม่ได้" in anon_what_if["answer"]


def test_database_inspector_and_sqlite_tables_match_architecture():
    client = TestClient(app)
    page = client.get("/db")
    assert page.status_code == 200
    assert "COCO-RISE Database Inspector" in page.text
    assert "ChromaDB" in page.text

    res = client.get("/v1/db/overview?table=fund_facts&only_verified=1")
    assert res.status_code == 200
    body = res.json()
    assert body["counts"]["document_registry"] == 4
    assert body["counts"]["document_chunks"] > 0
    assert body["counts"]["fund_facts"] > 0
    assert body["counts"]["eligibility_rules"] >= 4
    assert body["counts"]["eval_dataset"] > 0
    assert body["counts"]["applicant_profiles"] >= 5
    assert "ChromaDB" in body["engine"]
    assert body["vector_db"]["collection"] == "coco_rise_chunks"
    assert "chroma_id" in body["postgres_schema"]
    assert all(r["verified"] == 1 for r in body["rows"])


def test_chromadb_vector_store_sync_and_cosine_search(tmp_path):
    from app.infrastructure.chroma import vectors as chroma_vectors

    sample_chunks = [
        {
            "id": "ch_1",
            "doc": "หลักเกณฑ์กลาง",
            "fiscal_year": None,
            "clause": "ข้อ 4.1",
            "page": 2,
            "text": "ทุนวิจัยสถาบัน เพดานไม่เกิน 25,000 บาท",
            "status": "active",
            "verified": True,
        },
        {
            "id": "ch_2",
            "doc": "หลักเกณฑ์กลาง",
            "fiscal_year": None,
            "clause": "ข้อ 4.2",
            "page": 3,
            "text": "ทุนวิจัยในชั้นเรียน เพดานไม่เกิน 15,000 บาท",
            "status": "active",
            "verified": True,
        },
    ]
    vecs = {
        "ch_1": [1.0, 0.0, 0.0, 0.0],
        "ch_2": [0.0, 1.0, 0.0, 0.0],
    }
    chroma_dir = tmp_path / "chroma_test"
    count = chroma_vectors.sync_corpus_chunks(sample_chunks, vecs, "bge-m3:latest", chroma_dir=chroma_dir)
    assert count == 2

    chunks_by_id = {c["id"]: c for c in sample_chunks}
    hits = chroma_vectors.search_corpus_chunks([0.99, 0.01, 0.0, 0.0], chunks_by_id, top_k=2, chroma_dir=chroma_dir)
    assert hits is not None and len(hits) == 2
    assert hits[0][1]["id"] == "ch_1"
    assert hits[0][0] > 0.95
