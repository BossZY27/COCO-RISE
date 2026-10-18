"""Tests for the parts that decide what the answer says.

The rule of this suite: nothing here writes to data/. The behaviour worth pinning down is the logic that
picks a fund, picks a row, judges eligibility and rejects an answer, and all of that is readable
without touching the files a reviewer owns.

    pip install pytest
    python -m pytest tests -q
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import answers, audit, identity, limits, llm, pipeline, review, store, understand
from app.main import app


# ---------------------------------------------------------------- input limits
def test_blank_question_is_refused():
    text, refusal = limits.clean_question("   ")
    assert text == "" and refusal


def test_overlong_question_is_refused():
    _, refusal = limits.clean_question("ก" * (limits.MAX_QUESTION_CHARS + 1))
    assert refusal and str(limits.MAX_QUESTION_CHARS) in refusal


def test_control_characters_are_stripped():
    text, refusal = limits.clean_question("ทุน\x00วิจัย\x07สถาบัน")
    assert refusal is None and "\x00" not in text and "\x07" not in text


def test_history_is_capped_to_recent_turns():
    kept = limits.clean_history([f"คำถามที่ {i}" for i in range(30)])
    assert len(kept) == limits.MAX_HISTORY_TURNS
    assert kept[-1] == "คำถามที่ 29"


def test_rate_limit_lets_the_budget_through_then_blocks():
    user = "rate-test-user"
    results = [limits.take(user)[0] for _ in range(limits.RATE_LIMIT + 3)]
    assert results[:limits.RATE_LIMIT] == [True] * limits.RATE_LIMIT
    assert results[limits.RATE_LIMIT:] == [False, False, False]


# ---------------------------------------------------------------- who is asking
def test_header_identity_wins_over_the_request_body():
    user_id, profile = identity.resolve("P2", "P1")
    assert user_id == "P2" and profile["id"] == "P2"


def test_unknown_user_resolves_to_no_profile():
    _, profile = identity.resolve(None, "ไม่มีคนนี้")
    assert profile is None


def test_missing_fields_names_what_the_rules_need():
    assert set(identity.missing_fields(None)) == set(identity.REQUIRED)


def test_identity_contract_uses_the_department_field_read_by_eligibility_rules():
    assert "department" in identity.REQUIRED
    assert "faculty" not in identity.REQUIRED


# ---------------------------------------------------------------- reading the question
@pytest.mark.parametrize("question,intent", [
    ("ทุนวิจัยสถาบันปีนี้ให้ทุนละเท่าไร", "amount"),
    ("เพดานตามหลักเกณฑ์กลางของทุนวิจัยสถาบันคือเท่าไร", "ceiling"),
    ("พระขอได้มั้ย", "eligibility"),
    ("เงินสมทบหมายถึงอะไร", "definition"),
])
def test_intent_is_read_from_the_question(question, intent):
    assert pipeline.analyze(question)["intent"] == intent


def test_asking_a_funds_conditions_is_not_mistaken_for_an_amount_question():
    # Conditions have their own answer now (answers.details_answer): the rates, then where eligibility lives.
    assert pipeline.analyze("ทุนวิจัยสถาบันมีเงื่อนไขอะไรบ้าง")["intent"] == "details"


def test_asking_what_a_named_fund_is_gets_its_verified_category_not_a_money_answer():
    result = asyncio.run(pipeline.run("ทุนวิจัยสถาบันคืออะไร"))
    assert result["verify"]["ok"]
    assert "ทุนส่งเสริมการวิจัย" in result["answer"]
    assert "วงเงิน" not in result["answer"]
    assert any("หลักเกณฑ์กลาง" in c for c in result["citations"])


def test_application_process_question_does_not_turn_into_an_amount_answer():
    result = asyncio.run(pipeline.run("ทุนวิจัยสถาบันสมัครยังไง ต้องใช้เอกสารอะไร"))
    assert result["slots"]["intent"] == "application"
    assert "ยังไม่มีข้อมูล" in result["answer"]
    assert "เอกสาร" in result["answer"]
    assert "เพดาน" not in result["answer"]


def test_fund_without_a_reviewed_eligibility_rule_does_not_answer_with_money():
    result = asyncio.run(pipeline.run("ทุนทำวิจัยในต่างประเทศขอได้ไหม"))
    assert result["slots"]["intent"] == "eligibility"
    assert result["ruling"]["verdict"] == "no_rule"
    assert "ยังสรุปสิทธิ์ไม่ได้" in result["answer"]
    assert "เพดาน" not in result["answer"]


def test_named_eligibility_status_cites_the_applicant_definition():
    result = asyncio.run(pipeline.run("ฉันขอทุนวิจัยสถาบันได้ไหม"))

    if "คุณอยู่ในกลุ่มผู้ขอรับทุน" in result["answer"]:
        assert result["citations"]
        assert any("หลักเกณฑ์กลาง" in citation for citation in result["citations"])


@pytest.mark.parametrize("question", [
    "ทุนวิจัยสถาบันคืออะไร",
    "ทุนวิจัยสถาบันสมัครยังไง ต้องใช้เอกสารอะไร",
    "ทุนทำวิจัยในต่างประเทศขอได้ไหม",
])
def test_information_answers_match_between_regular_and_streaming_routes(question):
    async def stream_result():
        events = [event async for event in pipeline.run_stream(question)]
        return events[-1]

    regular = asyncio.run(pipeline.run(question))
    streamed = asyncio.run(stream_result())
    assert streamed["type"] == "done"
    assert streamed["answer"] == regular["answer"]
    assert streamed["citations"] == regular["citations"]
    assert streamed["verify"]["ok"] and regular["verify"]["ok"]


def test_only_installed_answer_models_are_offered(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [
                {"name": "thai-qwen:latest", "size": 100, "details": {"family": "qwen35"}},
                {"name": "bge-m3:latest", "size": 50, "details": {"family": "bert"}},
            ]}

    monkeypatch.setattr(llm.httpx, "get", lambda *_args, **_kwargs: Response())
    assert llm.installed_chat_models() == [{"name": "thai-qwen:latest", "size": 100}]


def test_model_api_reports_verified_template_mode_and_rejects_selection(monkeypatch):
    monkeypatch.setattr(limits, "take", lambda _user: (True, 0))
    client = TestClient(app)
    listing = client.get("/v1/models")
    assert listing.status_code == 200
    assert listing.json()["enabled"] is False
    assert listing.json()["models"] == []
    assert listing.json()["answer_mode"] == "verified_template"
    for route in ("/v1/query", "/v1/query/stream"):
        response = client.post(route, json={"question": "มีทุนอะไรบ้าง", "model": "not-installed"})
        assert response.status_code == 400


def test_safe_answer_mode_does_not_consult_ollama_for_model_selection(monkeypatch):
    from fastapi import HTTPException
    from app.main import _selected_model

    def down():
        raise RuntimeError("ollama is not running")

    monkeypatch.setattr(llm, "installed_chat_models", down)
    assert _selected_model(None) is None
    with pytest.raises(HTTPException) as refused:
        _selected_model("thai-qwen:latest")
    assert refused.value.status_code == 400


def test_explicit_model_is_rejected_before_regular_or_streaming_pipeline(monkeypatch):
    monkeypatch.setattr(limits, "take", lambda _user: (True, 0))
    seen = []

    async def fake_run(_question, _user, _history, model):
        seen.append(("regular", model))
        return {"selected_model": model}

    async def fake_stream(_question, _user, _history, model):
        seen.append(("stream", model))
        yield {"type": "done", "selected_model": model}

    monkeypatch.setattr(pipeline, "run", fake_run)
    monkeypatch.setattr(pipeline, "run_stream", fake_stream)
    client = TestClient(app)
    body = {"question": "มีทุนอะไรบ้าง", "model": "qwen3:4b"}
    assert client.post("/v1/query", json=body).status_code == 400
    assert client.post("/v1/query/stream", json=body).status_code == 400
    assert seen == []


def test_amount_word_beats_eligibility_word():
    # "ขอได้เท่าไร" contains both; the reader wants the money, not a yes or no
    assert pipeline.analyze("ทุนวิจัยสถาบัน ขอได้เท่าไร")["intent"] == "amount"


def test_a_vague_question_names_no_fund():
    for question in ["ทุนวิจัยนี้คืออะไร", "ฉันสามารถขอได้มั้ย", "สวัสดี"]:
        assert pipeline.analyze(question)["fund"] is None, question


def test_a_named_fund_is_matched():
    assert pipeline.analyze("ขอทุนวิจัยในชั้นเรียนได้ไหม")["fund"] == "ทุนวิจัยในชั้นเรียน"


def test_this_grant_takes_the_name_from_the_conversation():
    slots = pipeline.analyze("ทุนนี้ขอได้เท่าไร", ["ทุนวิจัยสถาบันปีนี้ให้ทุนละเท่าไร"])
    assert slots["fund"] == "ทุนวิจัยสถาบัน"


def test_history_is_scanned_past_a_vague_turn():
    history = ["ทุนวิจัยสถาบันปีนี้ให้ทุนละเท่าไร", "แล้วยังไงต่อ"]
    assert pipeline.analyze("ขอได้เท่าไร", history)["fund"] == "ทุนวิจัยสถาบัน"


@pytest.mark.parametrize("question", [
    "วันนี้อากาศเป็นยังไง",
    "ช่วยเขียนโค้ดให้หน่อย",
])
def test_unrelated_message_does_not_inherit_the_previous_fund_or_intent(question):
    slots = pipeline.analyze(question, ["ทุนวิจัยสถาบันได้เท่าไร"])

    assert slots["fund"] is None
    assert slots["intent"] == "unknown"


def test_a_reference_with_nothing_to_refer_to_is_flagged():
    # "ทุนนี้" as the very first question points at a turn that does not exist
    slots = pipeline.analyze("ทุนนี้คืออะไร")
    assert slots["fund"] is None and slots["unresolved_reference"] is True


def test_a_reference_the_conversation_resolves_is_not_flagged():
    slots = pipeline.analyze("ทุนนี้คืออะไร", ["ทุนวิจัยสถาบันให้เท่าไร"])
    assert slots["fund"] == "ทุนวิจัยสถาบัน" and slots["unresolved_reference"] is False


def test_a_question_naming_no_fund_is_not_a_dangling_reference():
    # nothing to resolve is different from a reference that failed to resolve
    assert pipeline.analyze("สวัสดี")["unresolved_reference"] is False


def test_the_current_question_wins_over_history():
    slots = pipeline.analyze("แล้วทุนวิจัยในชั้นเรียนล่ะ", ["ทุนวิจัยสถาบันให้เท่าไร"])
    assert slots["fund"] == "ทุนวิจัยในชั้นเรียน"


# ---------------------------------------------------------------- picking the right row
def test_ceiling_and_allocated_are_separate_rows():
    ceiling = store.fact_lookup("ทุนวิจัยความเป็นเลิศ", "ชั้นต้น", None, "ceiling")
    allocated = store.fact_lookup("ทุนวิจัยความเป็นเลิศ", "ชั้นต้น", 2569, "allocated")
    assert ceiling and allocated
    assert ceiling[0]["amount"] != allocated[0]["amount"]


def test_the_tier_row_is_preferred_when_a_tier_was_asked():
    rows = store.fact_lookup("ทุนวิจัยความเป็นเลิศ", "ชั้นกลาง", None, "ceiling")
    assert rows[0]["tier"] == "ชั้นกลาง"


def test_a_year_with_no_announcement_returns_nothing_allocated():
    assert store.fact_lookup("ทุนวิจัยความเป็นเลิศ", None, 2500, "allocated") == []


def test_definitions_are_read_out_of_the_criteria():
    applicant = store.define("ผู้ขอรับทุน")
    assert applicant and "วิทยาลัยการคอมพิวเตอร์" in applicant["text"]


def test_a_year_independent_rule_applies_to_any_year():
    rules = store.rules_for("เงินรางวัลการตีพิมพ์ ผู้ประพันธ์บรรณกิจ", 2569)
    assert any(r["fiscal_year"] is None for r in rules)


# ---------------------------------------------------------------- the catalogue and suggestions
def test_asking_what_funds_exist_is_its_own_intent():
    for question in ["มีทุนอะไรบ้าง", "ทุนมีกี่ประเภท", "อยากรู้ว่ามีกองทุนอะไร"]:
        assert pipeline.analyze(question)["intent"] == "list", question


def test_catalogue_question_skips_retrieval_that_cannot_change_its_answer(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("The verified catalogue answer does not need search retrieval")

    monkeypatch.setattr(store, "keyword_search", unexpected)
    monkeypatch.setattr(pipeline, "_semantic", unexpected)
    monkeypatch.setattr(store, "fact_lookup", unexpected)

    evidence = asyncio.run(pipeline.retrieve(pipeline.analyze("มีทุนอะไรบ้าง")))

    assert evidence["chunks"] == []
    assert evidence["facts"] == []
    assert evidence["semantic_by"] == "not_needed"


@pytest.mark.parametrize("question", [
    "ทุนมีตัวไหนบ้าง",
    "ทุนแต่ละตัวคืออะไรบ้าง",
    "ทุนเเต่ละตัวคืออะไรบ้าง",
])
def test_real_catalogue_questions_do_not_ask_for_one_fund(question):
    result = asyncio.run(pipeline.run(question))
    assert result["slots"]["intent"] == "list"
    assert "ทุนส่งเสริมการวิจัย" in result["answer"]
    assert "ทุนอุดหนุนกิจกรรม" in result["answer"]
    assert result["citations"] and result["verify"]["ok"]


@pytest.mark.parametrize("question", [
    "ฉันขอทุนอะไรได้บ้าง",
    "ตอนนี้ฉันขอทุนอะไรได้บ้าง",
    "ผมขอทุนไหนได้บ้าง",
    "ฉันจะขอทุนวิจัยตัวไหนได้บ้าง",
    "ฉันสามารถขอทุนวิจัยอะไรได้บ้าง",
    "ฉันมีสิทธิ์ขอทุนอะไรบ้าง",
    "ทุนอะไรบ้างที่ฉันขอได้",
    "ฉันสมัครทุนอะไรได้บ้าง",
    "ฉันสามารถสมัครทุนอะไรได้บ้าง",
    "ฉันสามารถขอทุนอะไรบ้าง",
    "ผมสามารถสมัครทุนอะไรบ้าง",
    "ฉันยื่นทุนอะไรได้บ้าง",
    "ฉันยื่นขอทุนอะไรได้บ้าง",
    "ทุนไหนที่ฉันขอได้บ้าง",
])
def test_personal_grant_eligibility_does_not_present_all_funds_as_eligible(question):
    result = asyncio.run(pipeline.run(question))
    assert result["slots"]["intent"] == "eligibility"
    assert result["slots"]["fund"] is None
    assert "ยังสรุปไม่ได้ว่าคุณขอทุนใดได้บ้าง" in result["answer"]
    assert "ผู้ขอรับทุน" in result["answer"]
    assert "เงื่อนไขของทุนแต่ละตัว" in result["answer"]
    assert "มี 12 รายการ" not in result["answer"]
    assert result["citations"] and result["verify"]["ok"]


def test_broad_eligibility_never_allows_a_model_to_guess_one_fund():
    slots = pipeline.analyze("ฉันขอทุนอะไรได้บ้าง")

    assert slots["fund"] is None
    assert slots["intent"] == "eligibility"
    assert slots["method"] == "rules"


@pytest.mark.parametrize("question", ["ฉันขอทุนอะไรได้บ้าง", "ทุนตัวไหนที่ฉันขอได้บ้าง"])
def test_broad_eligibility_gives_reviewed_fund_information_before_personal_check(question):
    result = asyncio.run(pipeline.run(question))
    assert result["slots"]["intent"] == "eligibility"
    assert "ทุนส่งเสริมการวิจัย" in result["answer"]
    assert "ทุนอุดหนุนกิจกรรม" in result["answer"]
    assert "ยังสรุปไม่ได้ว่าคุณขอทุนใดได้บ้าง" in result["answer"]
    assert "ระบุชื่อทุน" in result["answer"]
    assert not all(row["fund"] in result["answer"] for row in store.fund_catalogue())
    assert "บาท" not in result["answer"]
    assert result["citations"] and result["verify"]["ok"]


def test_recommendation_request_explains_information_only_scope():
    result = asyncio.run(pipeline.run("ฉันเหมาะกับทุนไหนบ้าง", history=["มีทุนอะไรบ้าง"]))
    assert result["slots"]["intent"] == "recommendation"
    assert "ยังไม่เปิดระบบแนะนำทุน" in result["answer"]
    assert "ระบุชื่อทุน" in result["answer"]
    assert not all(row["fund"] in result["answer"] for row in store.fund_catalogue())
    assert not result["citations"] and result["verify"]["ok"]


@pytest.mark.parametrize("question", [
    "แนะนำทุนสำหรับตีพิมพ์บทความ",
    "ช่วยแนะนำทุนไปนำเสนอผลงานที่ต่างประเทศ",
    "แนะนำทุนทำวิจัยในชั้นเรียน",
    "แนะนำทุนเกี่ยวกับการสอน",
])
def test_recommendation_request_does_not_pick_funds_even_with_a_work_description(question):
    first = asyncio.run(pipeline.run(question, persona_id="P1"))
    second = asyncio.run(pipeline.run(question, persona_id="P5"))
    for result in (first, second):
        assert result["slots"]["intent"] == "recommendation"
        assert "ยังไม่เปิดระบบแนะนำทุน" in result["answer"]
        assert not any(row["fund"] in result["answer"] for row in store.fund_catalogue())
        assert not result["citations"] and result["verify"]["ok"]
        assert result["answer_source"] == "template"
    assert first["answer"] == second["answer"]


@pytest.mark.parametrize("question", ["แนะนำทุนสำหรับซื้อรถ", "แนะนำทุนไปเที่ยวญี่ปุ่น"])
def test_recommendation_does_not_guess_for_unsupported_purpose(question):
    result = asyncio.run(pipeline.run(question))
    assert result["slots"]["intent"] == "recommendation"
    assert "ยังไม่เปิดระบบแนะนำทุน" in result["answer"]
    assert not any(row["fund"] in result["answer"] for row in store.fund_catalogue())
    assert result["verify"]["ok"]


def test_unsupported_purchase_question_says_uncertain_without_guessing_a_fund():
    result = asyncio.run(pipeline.run("ขอทุนซื้อคอมพิวเตอร์ได้ไหม"))
    assert result["slots"]["intent"] == "eligibility"
    assert "ไม่แน่ใจ" in result["answer"]
    assert not any(row["fund"] in result["answer"] for row in store.fund_catalogue())
    assert result["verify"]["ok"]


@pytest.mark.parametrize("question", ["อยากไปนำเสนองานที่ต่างประเทศ", "อยากตีพิมพ์วารสารนานาชาติ"])
def test_unnamed_purpose_does_not_select_a_fund(question):
    result = asyncio.run(pipeline.run(question))
    assert not any(row["fund"] in result["answer"] for row in store.fund_catalogue())
    assert "มีทุนอะไรบ้าง" in result["answer"]
    assert result["verify"]["ok"]


@pytest.mark.parametrize("question", [
    "ฉันขอทุนอะไรมาแล้วบ้าง",
    "ฉันสมัครทุนอะไรไปแล้วบ้าง",
    "สมัครทุนอะไรต้องใช้เอกสารอะไรบ้าง",
    "ขอทุนอะไรมีวงเงินเท่าไร",
])
def test_history_application_and_amount_questions_are_not_broad_eligibility(question):
    assert pipeline.BROAD_ELIGIBILITY_RE.search(question) is None


@pytest.mark.parametrize("question", [
    "ฉันเคยขอทุนอะไรบ้าง",
    "ฉันเคยขอทุนไหนบ้าง",
    "ฉันขอทุนอะไรมาแล้วบ้าง",
    "ฉันสมัครทุนอะไรไปแล้วบ้าง",
    "ฉันได้รับทุนอะไรบ้าง",
    "ทุนอะไรบ้างที่ฉันเคยขอมาแล้ว",
    "ฉันเคยขอทุนวิจัยสถาบันหรือยัง",
    "ฉันได้รับทุนวิจัยสถาบันแล้วหรือยัง",
    "ฉันเคยได้รับเงินรางวัลการตีพิมพ์ ผู้แต่งชื่อแรกหรือยัง",
    "ฉันได้รับทุนวิจัยสถาบันไหม",
    "ฉันได้รับทุนวิจัยสถาบันหรือเปล่า",
    "ฉันได้ทุนวิจัยสถาบันแล้วหรือยัง",
    "ฉันเคยได้เงินทุนวิจัยสถาบันหรือยัง",
])
def test_personal_history_questions_do_not_return_the_fund_catalogue(question):
    result = asyncio.run(pipeline.run(question))
    assert result["slots"]["intent"] == "history"
    assert "ยังไม่ได้เชื่อมทะเบียนประวัติ" in result["answer"]
    assert "มี 12 รายการ" not in result["answer"]
    assert "มี 16 รายการ" not in result["answer"]
    assert result["verify"]["ok"]


def test_broad_history_question_does_not_inherit_a_previous_fund():
    slots = pipeline.analyze("ฉันเคยขอทุนอะไรบ้าง", history=["ทุนวิจัยสถาบันมีเพดานเท่าไร"])
    assert slots["intent"] == "history"
    assert slots["fund"] is None


def test_previous_grant_followed_by_a_new_eligibility_question_is_not_history():
    slots = pipeline.analyze("เคยได้ทุนวิจัยสถาบันแล้ว ขอทุนวิจัยในชั้นเรียนได้ไหม")
    assert slots["intent"] == "hypothetical"
    assert slots["funds"] == ["ทุนวิจัยสถาบัน", "ทุนวิจัยในชั้นเรียน"]


@pytest.mark.parametrize("question", [
    "ถ้าเคยขอทุนวิจัยสถาบัน จะขอซ้ำได้ไหม",
    "ถ้าเคยขอทุนวิจัยสถาบัน ขออีกได้ไหม",
])
def test_same_fund_repeat_eligibility_is_not_treated_as_a_history_lookup(question):
    slots = pipeline.analyze(question)
    assert slots["intent"] == "eligibility"
    assert slots["fund"] == "ทุนวิจัยสถาบัน"


def test_same_turn_fund_name_wins_over_this_fund_anaphora():
    result = asyncio.run(pipeline.run("ถ้าได้ทุนวิจัยสถาบันแล้ว ขอทุนนี้อีกได้ไหม"))
    assert result["slots"]["intent"] == "eligibility"
    assert result["slots"]["fund"] == "ทุนวิจัยสถาบัน"
    assert result["slots"]["unresolved_reference"] is False
    assert "ระบบยังไม่แน่ใจว่าคำถามนี้หมายถึงทุนใด" not in result["answer"]
    assert result["verify"]["ok"]


def test_general_who_can_apply_question_does_not_use_a_personal_disclaimer():
    result = asyncio.run(pipeline.run("ใครมีสิทธิ์ขอทุนอะไรบ้าง"))
    assert result["slots"]["intent"] == "eligibility"
    assert "ผู้ขอรับทุน" in result["answer"]
    assert "ยังสรุปไม่ได้ว่าคุณขอทุนใดได้บ้าง" not in result["answer"]
    assert "ตรวจสิทธิ์ของคุณ" not in result["answer"]
    assert result["verify"]["ok"]


def test_broad_eligibility_question_does_not_inherit_a_previous_fund():
    slots = pipeline.analyze("ฉันขอทุนอะไรได้บ้าง", history=["ทุนวิจัยสถาบันมีเพดานเท่าไร"])
    assert slots["fund"] is None
    assert slots["intent"] == "eligibility"


def test_currently_open_funds_are_not_confused_with_the_criteria_catalogue():
    result = asyncio.run(pipeline.run("มีทุนอะไรบ้างที่เปิดตอนนี้"))
    assert result["slots"]["intent"] == "open_now"
    assert "ยังบอกไม่ได้ว่าทุนใดเปิดรับอยู่ตอนนี้" in result["answer"]
    assert "มี 12 รายการ" not in result["answer"]
    assert result["verify"]["ok"]


def test_recommendation_after_catalogue_uses_context_without_guessing_eligibility():
    result = asyncio.run(pipeline.run(
        "แนะนำทุนที่น่าจะเข้ากับฉันหน่อย", history=["มีทุนอะไรบ้างที่เปิดตอนนี้"]
    ))
    assert result["slots"]["intent"] == "recommendation"
    assert result["slots"]["catalogue_context"] is True
    assert "จากคำถามก่อนหน้าที่คุณถามถึงรายการทุน" in result["answer"]
    assert "ยังไม่เปิดระบบแนะนำทุน" in result["answer"]
    assert "ระบุชื่อทุน" in result["answer"]
    assert "มี 12 รายการ" not in result["answer"]
    assert result["verify"]["ok"]


@pytest.mark.parametrize("question,history", [
    ("บอกข้อมูลทุนแต่ละตัวมาหน่อยขอทั้งหมดนะ", []),
    ("บอกข้อมูลทุนแต่ละตัว", ["มีทุนอะไรแนะนำฉันบ้าง"]),
    ("ทุนตัวไหนทำหน้าที่และทำไรได้บ้าง", ["มีทุนอะไรแนะนำฉันบ้าง"]),
    ("ลงไปดูมาหน่อยแต่ละตัว", ["มีทุนอะไรบ้าง"]),
    ("ทุนแต่ละตัวยังไง", ["มีทุนอะไรบ้าง"]),
    ("ขอรายละเอียดทุกทุน", []),
])
def test_all_fund_details_answer_uses_every_reviewed_catalogue_row(question, history):
    result = asyncio.run(pipeline.run(question, history=history))
    assert result["slots"]["intent"] == "catalogue_details"
    assert "เพดานตามหลักเกณฑ์กลาง ไม่ใช่วงเงินจัดสรรรายปี" in result["answer"]
    assert "ยังไม่ใช่การยืนยันสิทธิ์หรือสถานะเปิดรับ" in result["answer"]
    assert all(row["fund"] in result["answer"] for row in store.fund_catalogue())
    assert result["citations"] and result["verify"]["ok"]


def test_all_fund_details_do_not_use_draft_annual_amounts():
    slots = pipeline.analyze("บอกข้อมูลทุนแต่ละตัว")
    evidence = asyncio.run(pipeline.retrieve(slots))
    answer, citations, used = pipeline.compose_answer(slots, evidence, {"verdict": "no_rule"})
    assert answer and citations
    assert used and all(row["amount_type"] == "ceiling" and store.is_verified(row) for row in used)
    assert not any("ประกาศ 42/2568" in citation for citation in citations)


def test_all_fund_details_do_not_override_named_or_eligibility_questions():
    assert pipeline.analyze("มีทุนอะไรบ้าง")["intent"] == "list"
    assert pipeline.analyze("ฉันขอทุนอะไรได้บ้าง")["intent"] == "eligibility"
    assert pipeline.analyze("ทุนวิจัยสถาบันมีรายละเอียดอะไรบ้าง")["intent"] == "details"
    assert pipeline.analyze("ทุนทั้งหมดเปิดรับสมัครยังไง")["intent"] == "application"


def test_explain_named_fund_and_clarifying_details_keep_the_same_fund():
    first = "อธิบาย ทุนวิจัยในชั้นเรียน หน่อย"
    first_result = asyncio.run(pipeline.run(first))
    followup = asyncio.run(pipeline.run("ฉันหมายถึงขอรายละเอียด", history=[first]))

    for result in (first_result, followup):
        assert result["slots"]["fund"] == "ทุนวิจัยในชั้นเรียน"
        assert result["slots"]["intent"] == "details"
        assert "10,000" in result["answer"]
        assert result["citations"] and result["verify"]["ok"]


def test_details_followup_does_not_inherit_a_fund_from_an_unrelated_catalogue_question():
    result = asyncio.run(pipeline.run("ฉันหมายถึงขอรายละเอียด", history=["มีทุนอะไรบ้าง"]))
    assert result["slots"]["fund"] is None
    assert "ทุนวิจัยในชั้นเรียนมีเพดาน" not in result["answer"]


@pytest.mark.parametrize("question", [
    "มีทุนตัวไหนบ้าง",
    "มีทุนไหนบ้าง",
    "ทุนตัวไหนบ้าง",
    "มีทุนอะไรให้เลือกบ้าง",
    "ช่วยบอกหน่อยว่ามีทุนตัวใดบ้าง",
    "มีทุนวิจัยอะไรบ้าง",
    "ทุนวิจัยมีอะไรบ้าง",
])
def test_fund_catalogue_paraphrases_are_list_questions(question):
    slots = pipeline.analyze(question)
    assert slots["intent"] == "list"
    evidence = {"facts": [], "chunks": []}
    answer, citations, used = pipeline.compose_answer(slots, evidence, {})
    assert "ทุนส่งเสริมการวิจัย" in answer
    assert "ทุนอุดหนุนกิจกรรม" in answer
    assert citations and not used
    assert pipeline.verify(answer, citations, used, evidence, slots, check_numbers=False)["ok"]


@pytest.mark.parametrize("question, intent", [
    ("ทุนวิจัยสถาบันมีเงื่อนไขอะไรบ้าง", "details"),
    ("ทุนวิจัยสถาบันขอได้เท่าไร", "amount"),
    ("ทุนวิจัยสถาบันคืออะไร", "definition"),
])
def test_catalogue_paraphrases_do_not_swallow_fund_specific_questions(question, intent):
    assert pipeline.analyze(question)["intent"] == intent


def test_the_catalogue_covers_both_headings_of_the_annex():
    groups = {row["group"] for row in store.fund_catalogue()}
    assert groups == {"ทุนส่งเสริมการวิจัย", "ทุนอุดหนุนกิจกรรม"}
    assert len(store.fund_catalogue()) >= 25


def test_a_fund_is_recognised_despite_different_wording():
    chunk = [{"clause": "2.2", "text": "2.2 เงินรางวัลการตีพิมพ์เผยแพร่ผลงานวิจัย (กรณีเป็นผู้แต่งชื่อแรก)"}]
    assert "เงินรางวัลการตีพิมพ์ ผู้แต่งชื่อแรก" in store.funds_mentioned_in(chunk)


def test_a_question_the_documents_cover_finds_a_supporting_clause():
    chunks = [{"clause": "2.11", "text": "2.11 ทุนนำเสนอผลงานทางวิชาการ ต่างประเทศ ตามที่จ่ายจริงแต่ไม่เกิน 80,000 บาท"}]
    assert store.relevant_fund_clauses("อยากไปนำเสนองานที่ต่างประเทศ", chunks)


def test_a_question_the_documents_do_not_cover_finds_nothing():
    chunks = [{"clause": "2.11", "text": "2.11 ทุนนำเสนอผลงานทางวิชาการ ต่างประเทศ ตามที่จ่ายจริง"}]
    assert store.relevant_fund_clauses("อยากได้ไปญี่ปุ่น", chunks) == []


def test_a_definitions_page_is_never_used_to_suggest_a_fund():
    chunks = [{"clause": "ทั้งหน้า", "text": "ทุนวิจัยสถาบัน ทุนวิจัยในชั้นเรียน ทุนวิจัยความเป็นเลิศ"}]
    assert store.relevant_fund_clauses("ทุนวิจัยสถาบัน", chunks) == []


# ---------------------------------------------------------------- eligibility
def _profile(**overrides) -> dict:
    base = {"id": "T1", "name": "ทดสอบ", "staff_type": "พนักงานมหาวิทยาลัย",
            "department": "วิทยาลัยการคอมพิวเตอร์", "grants_this_year": {}, "source": "test"}
    return {**base, **overrides}


@pytest.fixture
def approved_annual_rule_source(monkeypatch):
    # Simulate a reviewer-approved document only in memory; never write verified=true to data/.
    chunks = [{**c, "status": "active", "verified": True} if c["doc"] == "ประกาศ 42/2568" else c
              for c in store.CHUNKS]
    monkeypatch.setattr(store, "INGESTED_CHUNKS", chunks, raising=False)


def test_active_but_unverified_transcription_is_not_searchable():
    chunk = {"status": "active", "verified": False, "fiscal_year": 2569}
    assert not store._passes_filter(chunk, 2569)


def test_rules_from_pending_scan_are_not_used():
    assert not store.rules_for("ทุนวิจัยในชั้นเรียน", 2569)
    assert store.rules_for("เงินรางวัลการตีพิมพ์ ผู้ประพันธ์บรรณกิจ", 2569)


def test_classroom_research_rule_calls_one_grant_an_annual_total():
    rule = next(r for r in store.RULES if r["id"] == "r02")
    reason = next(c["fail_reason"] for c in rule["conditions"]
                  if c["field"].startswith("grants_this_year."))
    assert "จัดสรรทุนนี้ไม่เกิน 1 ทุนรวม" in reason
    assert "คนละ" not in reason


def test_a_matching_profile_is_eligible(approved_annual_rule_source):
    slots = pipeline.analyze("ขอทุนวิจัยในชั้นเรียนได้ไหม")
    assert pipeline.check_rules(_profile(), slots)["verdict"] == "eligible"


def test_the_wrong_faculty_is_not_eligible(approved_annual_rule_source):
    slots = pipeline.analyze("ขอทุนวิจัยในชั้นเรียนได้ไหม")
    ruling = pipeline.check_rules(_profile(department="คณะอื่น"), slots)
    assert ruling["verdict"] == "not_eligible" and ruling["reason"]


def test_one_grant_per_year_blocks_a_second_request(approved_annual_rule_source):
    slots = pipeline.analyze("ขอทุนวิจัยในชั้นเรียนอีกทุนได้ไหม")
    ruling = pipeline.check_rules(_profile(grants_this_year={"ทุนวิจัยในชั้นเรียน": 1}), slots)
    assert ruling["verdict"] == "not_eligible"


def test_a_missing_profile_gives_unknown_rather_than_a_refusal(approved_annual_rule_source):
    slots = pipeline.analyze("ขอทุนวิจัยในชั้นเรียนได้ไหม")
    ruling = pipeline.check_rules({"id": "T2", "name": "ว่าง", "grants_this_year": {}}, slots)
    assert ruling["verdict"] == "unknown"


def test_a_grant_held_elsewhere_blocks_the_one_that_excludes_it():
    blocker = "เงินรางวัลการตีพิมพ์ ผู้แต่งชื่อแรก"
    slots = pipeline.analyze("ขอเงินรางวัลการตีพิมพ์ ผู้ประพันธ์บรรณกิจ ได้ไหม")
    ruling = pipeline.check_rules(_profile(grants_this_year={blocker: 1}), slots)
    assert ruling["verdict"] == "not_eligible" and blocker in ruling["reason"]


# ---------------------------------------------------------------- the last gate
def _evidence(*texts) -> dict:
    return {"chunks": [{"id": f"c{i}", "text": t} for i, t in enumerate(texts)], "facts": []}


def test_a_number_absent_from_the_evidence_is_rejected():
    fact = {"amount": 25000, "total_cap": None, "max_grants_per_person": None,
            "fiscal_year": 2569, "amount_type": "allocated", "note": None}
    check = pipeline.verify("ทุนนี้ได้ 999,999 บาท", ["ประกาศ 42/2568 ข้อ 4"], [fact],
                            _evidence("ทุนวิจัยสถาบัน 25,000 บาท"), {"fiscal_year": 2569})
    assert not check["ok"] and any("999,999" in p for p in check["problems"])


def test_a_fact_without_an_explicit_verified_marker_is_rejected():
    fact = {"fund": "ทุนวิจัยสถาบัน", "amount": 25000, "total_cap": None,
            "max_grants_per_person": None, "fiscal_year": None,
            "amount_type": "ceiling", "note": None}

    check = pipeline.verify(
        "เพดานทุนวิจัยสถาบันไม่เกิน 25,000 บาท",
        ["หลักเกณฑ์กลาง บัญชีแนบท้าย หน้า 4"],
        [fact],
        _evidence("ทุนวิจัยสถาบัน 25,000 บาท"),
        {"fiscal_year": 2569},
    )

    assert not check["ok"]
    assert any("ยังไม่ผ่านการอนุมัติ" in problem for problem in check["problems"])


def test_a_chunk_without_an_explicit_verified_marker_is_untrusted():
    chunk = {"status": "active", "fiscal_year": None}

    assert not store._passes_filter(chunk, None)


def test_number_check_uses_whole_values_not_substrings():
    fact = {"fund": "ทุนวิจัยสถาบัน", "amount": 25000, "total_cap": None,
            "max_grants_per_person": None, "fiscal_year": None,
            "amount_type": "ceiling", "verified": True, "note": None}
    answer = "เพดานทุนวิจัยสถาบันไม่เกิน 25,000 บาท สมัครภายใน 25 วัน"

    check = pipeline.verify(answer, ["หลักเกณฑ์กลาง บัญชีแนบท้าย หน้า 4"], [fact],
                            _evidence("ทุนวิจัยสถาบัน 25,000 บาท"), {"fiscal_year": 2569})

    assert not check["ok"] and any("25" in p for p in check["problems"])


def test_a_number_present_in_the_evidence_with_cap_wording_passes():
    fact = {"amount": 25000, "total_cap": None, "max_grants_per_person": None,
            "fiscal_year": 2569, "amount_type": "allocated", "verified": True, "note": None}
    check = pipeline.verify("ทุนนี้ขอได้สูงสุด 25,000 บาท", ["ประกาศ 42/2568 ข้อ 4"], [fact],
                            _evidence("ทุนวิจัยสถาบัน 25,000 บาท"), {"fiscal_year": 2569})
    assert check["ok"], check["problems"]


def test_model_reasoning_preamble_is_not_shown_as_a_thai_answer():
    fact = {"fund": "ทุนวิจัยสถาบัน", "amount": 25000, "total_cap": None,
            "max_grants_per_person": None, "fiscal_year": None,
            "amount_type": "ceiling", "verified": True, "note": None}
    answer = "Okay, let's see. The user is asking about ทุนวิจัยสถาบัน. ข้อมูลคือ 25,000 บาท"
    check = pipeline.verify(answer, ["หลักเกณฑ์กลาง บัญชีแนบท้าย หน้า 4"], [fact],
                            _evidence("ทุนวิจัยสถาบัน 25,000 บาท"), {"fiscal_year": 2569})
    assert not check["ok"]


@pytest.mark.parametrize("answer", ["25,000 บาท", "เพดาน 25,000 บาท", "ทุนวิจัยสถาบัน 25,000 บาท"])
def test_a_ceiling_answer_must_name_the_fund_and_the_cap(answer):
    fact = {"fund": "ทุนวิจัยสถาบัน", "amount": 25000, "total_cap": None,
            "max_grants_per_person": None, "fiscal_year": None,
            "amount_type": "ceiling", "verified": True, "note": None}
    check = pipeline.verify(answer, ["หลักเกณฑ์กลาง บัญชีแนบท้าย หน้า 4"], [fact],
                            _evidence("ทุนวิจัยสถาบัน 25,000 บาท"), {"fiscal_year": 2569})
    assert not check["ok"]
    good = pipeline.verify("เพดานทุนวิจัยสถาบันไม่เกิน 25,000 บาท",
                           ["หลักเกณฑ์กลาง บัญชีแนบท้าย หน้า 4"], [fact],
                           _evidence("ทุนวิจัยสถาบัน 25,000 บาท"), {"fiscal_year": 2569})
    assert good["ok"]


def test_the_number_gate_can_be_waived_for_code_built_answers():
    # the fund list carries clause numbers such as 1.1 and 2.15 that no passage contains
    lenient = pipeline.verify("ทุนวิจัยสถาบัน (ข้อ 1.2)", ["หลักเกณฑ์กลาง บัญชีแนบท้าย หน้า 4"],
                              [], _evidence("ไม่มีตัวเลขใด"), {"fiscal_year": 2569}, check_numbers=False)
    strict = pipeline.verify("ทุนวิจัยสถาบัน (ข้อ 1.2)", ["หลักเกณฑ์กลาง บัญชีแนบท้าย หน้า 4"],
                             [], _evidence("ไม่มีตัวเลขใด"), {"fiscal_year": 2569})
    assert lenient["ok"] and not strict["ok"]


def test_an_answer_with_no_citation_is_rejected():
    fact = {"amount": 25000, "total_cap": None, "max_grants_per_person": None,
            "fiscal_year": 2569, "amount_type": "allocated", "note": None}
    check = pipeline.verify("ทุนนี้ได้ 25,000 บาท", [], [fact],
                            _evidence("25,000 บาท"), {"fiscal_year": 2569})
    assert not check["ok"]


def test_evidence_from_the_wrong_year_is_rejected():
    fact = {"amount": 25000, "total_cap": None, "max_grants_per_person": None,
            "fiscal_year": 2568, "amount_type": "allocated", "note": None}
    check = pipeline.verify("ทุนนี้ได้ 25,000 บาท", ["ประกาศ"], [fact],
                            _evidence("25,000 บาท"), {"fiscal_year": 2569})
    assert not check["ok"]


# ---------------------------------------------------------------- the record
def test_an_answer_that_drops_the_figure_is_rejected():
    # a model told only what to avoid once answered by leaving the amount out entirely
    fact = {"fund": "ทุนวิจัยสถาบัน", "amount": 25000, "total_cap": None,
            "max_grants_per_person": None, "fiscal_year": 2569, "amount_type": "allocated", "note": None}
    check = pipeline.verify("ข้อมูลยังไม่เพียงพอที่จะระบุวงเงิน", ["ประกาศ 42/2568 ข้อ 4"], [fact],
                            _evidence("ทุนวิจัยสถาบัน 25,000 บาท"), {"fiscal_year": 2569})
    assert not check["ok"] and any("25,000" in p for p in check["problems"])


def test_an_answer_naming_a_different_fund_is_rejected():
    fact = {"fund": "ทุนวิจัยสถาบัน", "amount": 25000, "total_cap": None,
            "max_grants_per_person": None, "fiscal_year": 2569, "amount_type": "allocated", "note": None}
    check = pipeline.verify("ทุนวิจัยในชั้นเรียน ได้ 25,000 บาท", ["ประกาศ 42/2568 ข้อ 4"], [fact],
                            _evidence("ทุนวิจัยสถาบัน 25,000 บาท"), {"fiscal_year": 2569})
    assert not check["ok"] and any("ทุนวิจัยในชั้นเรียน" in p for p in check["problems"])


def test_an_answer_naming_the_right_fund_passes():
    fact = {"fund": "ทุนวิจัยสถาบัน", "amount": 25000, "total_cap": None,
            "max_grants_per_person": None, "fiscal_year": 2569, "amount_type": "allocated",
            "verified": True, "note": None}
    check = pipeline.verify("ทุนวิจัยสถาบัน ขอได้สูงสุด 25,000 บาท", ["ประกาศ 42/2568 ข้อ 4"], [fact],
                            _evidence("ทุนวิจัยสถาบัน 25,000 บาท"), {"fiscal_year": 2569})
    assert check["ok"], check["problems"]


def test_cited_cross_fund_rule_may_name_the_grant_that_blocks_eligibility():
    slots = pipeline.analyze("ขอเงินรางวัลการตีพิมพ์ ผู้ประพันธ์บรรณกิจ ได้ไหม")
    fact = store.fact_lookup(slots["fund"], None, None, "ceiling")[0]
    evidence = {"facts": [fact], "chunks": [], "ceiling_fallback": True}
    profile = _profile(grants_this_year={"เงินรางวัลการตีพิมพ์ ผู้แต่งชื่อแรก": 1})
    ruling = pipeline.check_rules(profile, slots)
    answer, citations, used = pipeline.compose_answer(slots, evidence, ruling)

    assert ruling["verdict"] == "not_eligible"
    assert pipeline.verify(answer, citations, used, evidence, slots, ruling=ruling)["ok"]


def test_an_allocated_amount_is_worded_as_a_cap():
    # the announcement heads the column "วงเงินสูงสุดต่อทุน", so the answer must not promise the figure
    slots = pipeline.analyze("ทุนวิจัยสถาบันปีนี้ให้ทุนละเท่าไร")
    evidence = {"facts": store.fact_lookup("ทุนวิจัยสถาบัน", None, 2569, "allocated"),
                "chunks": [], "ceiling_fallback": False}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}
    answer, _, _ = pipeline.compose_answer(slots, evidence, ruling)
    assert "สูงสุด" in answer and "ได้ 25,000" not in answer


def test_annual_announcement_does_not_turn_grant_count_into_per_person_limit():
    # Announcement 42/2568, page 3: "20,000 บาท/ไม่เกิน 4 ทุน" is a fund-wide count.
    fact = next(f for f in store.FACTS if f["id"] == "f09")
    assert fact["max_grants_per_person"] is None
    assert fact["max_grants_per_year"] == 4
    slots = pipeline.analyze("ทุนวิจัยเพื่อพัฒนางานประจำปี 2569 ได้กี่ทุน")
    evidence = {"facts": [fact], "chunks": [], "ceiling_fallback": False}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}
    answer, _, _ = pipeline.compose_answer(slots, evidence, ruling)
    assert "ไม่เกิน 4 ทุน" in answer
    assert "คนละ" not in answer


def test_announcement_42_has_a_row_for_every_figure_in_its_table():
    # Pages 3-4 of announcement 42/2568 print 16 figures; the team's table carried 11. A question about
    # a missing tier fell back to the central ceiling and was told no announcement existed.
    expected = {
        ("ทุนสนับสนุนผู้เชี่ยวชาญต่างประเทศ", None): 200000,
        ("ทุนวิจัยสถาบัน", None): 25000,
        ("ทุนวิจัยเพื่อพัฒนางานประจำ", None): 20000,
        ("ทุนวิจัยในชั้นเรียน", None): 10000,
        ("ทุนวิจัยความเป็นเลิศ", "ชั้นต้น"): 100000,
        ("ทุนวิจัยความเป็นเลิศ", "ชั้นกลาง"): 150000,
        ("ทุนวิจัยความเป็นเลิศ", "ชั้นสูง"): 250000,
        ("ทุนนวัตกรรมความเป็นเลิศ", "ชั้นต้น"): 100000,
        ("ทุนนวัตกรรมความเป็นเลิศ", "ชั้นกลาง"): 150000,
        ("ทุนนวัตกรรมความเป็นเลิศ", "ชั้นสูง"): 250000,
        ("ทุนพัฒนาศูนย์วิจัย", None): 250000,
        ("ทุนฝึกอบรมนักวิจัยหลังปริญญาเอก", None): 400000,
        ("ทุนสนับสนุนการได้รับทุนวิจัยภายนอก", None): 100000,
        ("ทุนพัฒนากลุ่มวิจัยบูรณาการ", "ชั้นต้น"): 200000,
        ("ทุนพัฒนากลุ่มวิจัยบูรณาการ", "ชั้นกลาง"): 300000,
        ("ทุนพัฒนากลุ่มวิจัยบูรณาการ", "ชั้นสูง"): 500000,
    }
    rows = [f for f in store.FACTS
            if f["amount_type"] == "allocated" and f["source"]["doc"] == "ประกาศ 42/2568"]
    assert {(f["fund"], f["tier"]): f["amount"] for f in rows} == expected
    assert len(rows) == len(expected)


def test_external_grant_support_keeps_its_ten_percent_condition():
    # Item 1.9 grants at most 10% of the external grant and never more than 100,000 baht. Printing only
    # the 100,000 reads as a flat amount, which is a different promise from the one the document makes.
    fact = next(f for f in store.FACTS if f["fund"] == "ทุนสนับสนุนการได้รับทุนวิจัยภายนอก"
                and f["amount_type"] == "allocated")
    reviewed = {**fact, "verified": True}          # in memory only; data/ keeps the reviewer's flag
    slots = pipeline.analyze("ทุนสนับสนุนการได้รับทุนวิจัยภายนอกปี 2569 ได้เท่าไร")
    evidence = {"facts": [reviewed], "chunks": [], "ceiling_fallback": False}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}
    answer, citations, used = pipeline.compose_answer(slots, evidence, ruling)
    assert "ร้อยละ 10" in answer and "100,000" in answer
    assert pipeline.verify(answer, citations, used, evidence, slots, ruling=ruling)["ok"]


def test_missing_annual_row_is_not_reported_as_a_missing_announcement():
    # Holding no annual row says nothing about whether an announcement exists. The old wording,
    # "ยังไม่มีประกาศจัดสรร", told readers that 42/2568 did not exist for tiers it does list.
    slots = pipeline.analyze("ทุนนักวิจัยอาวุโสปีงบ 2569 ได้เท่าไร")
    evidence = {"facts": store.fact_lookup(slots["fund"], None, None, "ceiling"), "chunks": [],
                "ceiling_fallback": True}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}
    answer, citations, used = pipeline.compose_answer(slots, evidence, ruling)
    assert slots["fund"] == "ทุนนักวิจัยอาวุโส"
    assert "ยังไม่มีประกาศจัดสรร" not in answer
    assert "ยังไม่มีข้อมูลวงเงินจัดสรร" in answer and "เพดาน" in answer
    assert pipeline.verify(answer, citations, used, evidence, slots, ruling=ruling)["ok"]


def test_a_rate_note_may_name_the_fund_its_rate_follows():
    # c29's note says its rates follow ทุนนำเสนอผลงานทางวิชาการ. Quoted whole from the reviewed row, that
    # name is the document speaking; the stray-fund check refused the whole answer over it.
    slots = pipeline.analyze("ทุนนำเสนอผลงานนวัตกรรมและสิ่งประดิษฐ์ปี 2569 ได้เท่าไร")
    evidence = {"facts": store.fact_lookup(slots["fund"], None, None, "ceiling"), "chunks": [],
                "ceiling_fallback": True}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}
    answer, citations, used = pipeline.compose_answer(slots, evidence, ruling)
    assert "ทุนนำเสนอผลงานทางวิชาการ" in answer
    assert pipeline.verify(answer, citations, used, evidence, slots, ruling=ruling)["ok"]
    # outside the quoted note the other fund's name is still a stray
    draft = ("ระบบยังไม่มีข้อมูลวงเงินจัดสรร เพดานของทุนนำเสนอผลงานทางวิชาการ "
             "วงเงินรวมไม่เกิน 200,000 บาท")
    assert not pipeline.verify(draft, citations, used, evidence, slots, ruling=ruling)["ok"]


def test_all_fact_answers_skip_free_form_model_rephrasing(monkeypatch):
    # A number gate cannot prove arbitrary prose is supported. Keep all fact-bearing
    # answers on the deterministic template until the model emits typed claims.
    monkeypatch.setattr(pipeline.llm, "enabled", lambda: True)
    evidence = {"facts": [], "chunks": []}
    with_figures = next(f for f in store.FACTS if f["id"] == "c30")
    wording_only = next(f for f in store.FACTS if f["id"] == "c29")
    plain = next(f for f in store.FACTS if f["id"] == "c06a")
    assert not pipeline._model_may_rephrase([with_figures], evidence)
    assert not pipeline._model_may_rephrase([wording_only], evidence)
    assert not pipeline._model_may_rephrase([plain], evidence)


def test_free_form_model_rephrasing_is_disabled_for_fact_answers(monkeypatch):
    monkeypatch.setattr(pipeline.llm, "enabled", lambda: True)
    fact = next(f for f in store.FACTS if f["id"] == "c06a")

    assert not pipeline._model_may_rephrase([fact], {"chunks": [], "facts": [fact]})


def test_pending_notice_keeps_the_rate_note_of_the_ceiling_it_quotes():
    # While an announcement waits for review the answer falls back to the central ceiling. If that row has
    # a rate note, the note is part of the figure: c30 caps some destinations at 50,000 baht.
    ceiling = next(f for f in store.FACTS if f["id"] == "c30")
    slots = {"fund": ceiling["fund"], "tier": None, "fiscal_year": 2569, "intent": "amount",
             "year_assumed": False, "standalone_question": ceiling["fund"]}
    evidence = {"facts": [], "chunks": [], "pending_review": True}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}
    answer, citations, used = pipeline.compose_answer(slots, evidence, ruling)
    assert used == [ceiling] and "50,000" in answer
    assert pipeline.verify(answer, citations, used, evidence, slots, ruling=ruling)["ok"]


def test_unreviewed_allocated_fact_is_not_used_to_answer():
    # OCR-derived money stays unavailable until a person approves it in /admin.
    slots = pipeline.analyze("ทุนวิจัยความเป็นเลิศชั้นต้นปี 2569 ได้กี่บาท")
    evidence = asyncio.run(pipeline.retrieve(slots))
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}
    answer, citations, used = pipeline.compose_answer(slots, evidence, ruling)
    assert used and all(f["amount_type"] == "ceiling" and f["verified"] for f in used)
    assert "100,000" not in answer
    assert "รอ" in answer and "อนุมัติ" in answer
    assert "เพดาน" in answer and "วงเงินจัดสรร" in answer
    assert any("หลักเกณฑ์กลาง" in c for c in citations)
    assert pipeline.verify(answer, citations, used, evidence, slots, ruling=ruling)["ok"]


def test_pending_annual_amount_cannot_be_rephrased_as_an_approved_allocation():
    slots = pipeline.analyze("ทุนวิจัยความเป็นเลิศชั้นต้นปี 2569 ได้กี่บาท")
    evidence = asyncio.run(pipeline.retrieve(slots))
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}
    _, citations, used = pipeline.compose_answer(slots, evidence, ruling)
    misleading = "ทุนวิจัยความเป็นเลิศชั้นต้น ปีงบประมาณ 2569 ได้สูงสุด 150,000 บาท"
    assert not pipeline.verify(misleading, citations, used, evidence, slots, ruling=ruling)["ok"]


def test_pending_annual_amount_uses_immediate_verified_template_not_llm(monkeypatch):
    monkeypatch.setattr(pipeline.llm, "enabled", lambda: True)

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("pending annual data must not enter the model")

    monkeypatch.setattr(pipeline.llm, "chat", unexpected_model_call)
    result = asyncio.run(pipeline.run("ทุนวิจัยความเป็นเลิศชั้นต้นปี 2569 ได้กี่บาท"))
    assert result["answer_source"] == "template"
    assert result["verify"]["ok"]
    assert "llm" not in result["timings_ms"]


def test_eval_does_not_score_a_ceiling_as_a_reviewed_annual_amount():
    import eval as evaluation

    case = {"expect_amount": "25,000", "expect_doc": "ประกาศ 42/2568"}
    result = {"answer": "เพดานตามหลักเกณฑ์กลาง 25,000 บาท ไม่ใช่วงเงินจัดสรรรายปี",
              "citations": ["หลักเกณฑ์กลาง บัญชีแนบท้าย ข้อ 1.2 หน้า 4"]}
    assert not evaluation.amount_matches_expectation(case, result)


def test_general_eligibility_answer_does_not_infer_exclusion_from_a_definition():
    answer, citations = answers.who_may_apply("ใครขอทุนได้บ้าง", None)
    assert citations
    assert "ผู้ขอรับทุน" in answer
    assert "ไม่ได้" not in answer


def test_a_role_outside_the_definition_gets_the_condition_not_a_verdict():
    # The definition lists three employment types and no roles. For "พระขอได้มั้ย" the answer states what
    # it requires and what follows for someone without it, never "พระขอไม่ได้" as a fact about monks.
    answer, _ = answers.who_may_apply("พระขอได้มั้ย", None)
    assert "ผู้ขอรับทุนคือบุคลากร" in answer
    assert "ถ้าพระไม่ได้มีสถานะเป็นบุคลากร" in answer
    assert "พระขอไม่ได้" not in answer and "พระขอทุนไม่ได้" not in answer


def test_pending_scan_eligibility_question_explains_why_it_cannot_decide():
    slots = pipeline.analyze("ขอทุนวิจัยในชั้นเรียนได้ไหม")
    evidence = asyncio.run(pipeline.retrieve(slots))
    ruling = pipeline.check_rules(_profile(), slots)
    answer, _, used = pipeline.compose_answer(slots, evidence, ruling)
    assert not used and ruling["verdict"] == "no_rule"
    assert "ยังสรุปสิทธิ์ไม่ได้" in answer
    assert "วงเงิน" not in answer


def test_fund_list_does_not_promise_unapproved_annual_amounts():
    answer, _ = pipeline._catalogue_answer()
    assert "วงเงินจัดสรรรายปีที่เจ้าหน้าที่อนุมัติแล้ว" in answer
    assert "หากยังรอตรวจจะแจ้งให้ทราบ" in answer


def test_model_cannot_turn_an_allocated_cap_into_a_promised_payment():
    fact = {"fund": "ทุนวิจัยสถาบัน", "amount": 25000, "total_cap": None,
            "max_grants_per_person": None, "fiscal_year": 2569, "amount_type": "allocated",
            "verified": True, "note": None}
    check = pipeline.verify("ทุนวิจัยสถาบัน ได้ 25,000 บาท", ["ประกาศ 42/2568 ข้อ 4"], [fact],
                            _evidence("วงเงินสูงสุดต่อทุน 25,000 บาท"), {"fiscal_year": 2569})
    assert not check["ok"]


def test_model_cannot_hide_that_a_figure_is_unverified():
    fact = {"fund": "ทุนวิจัยสถาบัน", "amount": 25000, "total_cap": None,
            "max_grants_per_person": None, "fiscal_year": 2569, "amount_type": "allocated",
            "verified": False, "note": None}
    check = pipeline.verify("ทุนวิจัยสถาบัน ขอได้สูงสุด 25,000 บาท", ["ประกาศ 42/2568 ข้อ 4"],
                            [fact], _evidence("วงเงินสูงสุดต่อทุน 25,000 บาท"), {"fiscal_year": 2569})
    assert not check["ok"]


def test_model_cannot_drop_the_condition_a_rate_note_carries():
    # The figure alone is in the evidence, so every other check passes a draft that drops "ร้อยละ 10".
    # Comparing whole numbers matters: "10" is a substring of "100,000".
    fact = {"fund": "ทุนสนับสนุนการได้รับทุนวิจัยภายนอก", "amount": 100000, "total_cap": 300000,
            "max_grants_per_person": 1, "fiscal_year": 2569, "amount_type": "allocated",
            "verified": True, "note": "ไม่เกินร้อยละ 10 ของเงินทุนที่ได้รับจากภายนอก และไม่เกิน 100,000 บาทต่อทุน"}
    evidence = _evidence("ไม่เกิน 10% ของเงินทุนที่ได้รับจากภายนอก และไม่เกิน 100,000 บาท/ทุน")
    dropped = pipeline.verify("ทุนสนับสนุนการได้รับทุนวิจัยภายนอก ขอได้สูงสุด 100,000 บาทต่อทุน",
                              ["ประกาศ 42/2568 ข้อ 4"], [fact], evidence, {"fiscal_year": 2569})
    kept = pipeline.verify("ทุนสนับสนุนการได้รับทุนวิจัยภายนอก ขอได้ไม่เกินร้อยละ 10 ของทุนภายนอก "
                           "และสูงสุด 100,000 บาทต่อทุน", ["ประกาศ 42/2568 ข้อ 4"], [fact], evidence,
                           {"fiscal_year": 2569})
    assert not dropped["ok"]
    assert kept["ok"], kept["problems"]


def test_code_built_answer_with_an_unverified_warning_is_rejected():
    slots = pipeline.analyze("ทุนวิจัยสถาบันปีนี้ให้ทุนละเท่าไร")
    evidence = {"facts": store.fact_lookup("ทุนวิจัยสถาบัน", None, 2569, "allocated"),
                "chunks": [], "ceiling_fallback": False}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}
    answer, citations, used = pipeline.compose_answer(slots, evidence, ruling)

    assert not pipeline.verify(answer, citations, used, evidence, slots)["ok"]


def test_stream_does_not_publish_unverified_model_tokens(monkeypatch):
    fact = {"id": "test-fact", "fund": "ทุนวิจัยสถาบัน", "tier": None, "amount": 25000,
            "total_cap": None, "max_grants_per_person": None, "fiscal_year": 2569,
            "amount_type": "allocated", "verified": True, "note": None,
            "source": {"doc": "ประกาศ 42/2568", "clause": "ข้อ 4", "page": 1}}
    slots = {"fund": fact["fund"], "fiscal_year": 2569, "intent": "amount", "year_assumed": False}
    evidence = {"chunks": [], "facts": [fact], "keyword_ids": [], "semantic_ids": [],
                "semantic_by": "trigram", "ceiling_fallback": False}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}

    async def fake_prepare(*_):
        return store.PERSONAS["P1"], slots, evidence, ruling, "ทุนวิจัยสถาบัน ขอได้สูงสุด 25,000 บาท", ["ประกาศ 42/2568 ข้อ 4"], [fact]

    def fake_stream_chat(*_):
        yield "ทุนวิจัยสถาบัน ได้ 999,999 บาท"

    monkeypatch.setattr(pipeline, "_prepare", fake_prepare)
    monkeypatch.setattr(pipeline.llm, "enabled", lambda: True)
    monkeypatch.setattr(pipeline.llm, "stream_chat", fake_stream_chat)

    async def collect():
        return [event async for event in pipeline.run_stream("ทุนวิจัยสถาบันได้เท่าไร", model="qwen3:4b")]

    events = asyncio.run(collect())
    assert not any(event["type"] == "token" for event in events)
    assert any(event["type"] == "preview" and "25,000" in event["answer"] for event in events)
    assert events[-1]["answer"] == "ทุนวิจัยสถาบัน ขอได้สูงสุด 25,000 บาท"
    assert events[-1]["selected_model"] is None
    assert events[-1]["answer_source"] == "template"


def test_fallback_answers_do_not_wait_for_a_model_draft(monkeypatch):
    # A draft of a fallback answer rarely keeps the exact status wording the gate requires, so a reader
    # waited about twenty seconds for a draft that was then thrown away. The template is the answer.
    slots = pipeline.analyze("ทุนนักวิจัยอาวุโสปีงบ 2569 ได้เท่าไร")
    facts = store.fact_lookup(slots["fund"], None, None, "ceiling")
    evidence = {"chunks": [], "facts": facts, "keyword_ids": [], "semantic_ids": [],
                "semantic_by": "trigram", "ceiling_fallback": True}
    ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": []}
    template, citations, used = pipeline.compose_answer(slots, evidence, ruling)
    calls = []

    async def fake_prepare(*_):
        return store.PERSONAS["P1"], slots, evidence, ruling, template, citations, used

    def fake_chat(*_args, **_kwargs):
        calls.append("chat")
        return template

    def fake_stream_chat(*_args, **_kwargs):
        calls.append("stream")
        yield template

    monkeypatch.setattr(pipeline, "_prepare", fake_prepare)
    monkeypatch.setattr(pipeline.llm, "enabled", lambda: True)
    monkeypatch.setattr(pipeline.llm, "chat", fake_chat)
    monkeypatch.setattr(pipeline.llm, "stream_chat", fake_stream_chat)

    regular = asyncio.run(pipeline.run("ทุนนักวิจัยอาวุโสปีงบ 2569 ได้เท่าไร"))

    async def collect():
        return [event async for event in pipeline.run_stream("ทุนนักวิจัยอาวุโสปีงบ 2569 ได้เท่าไร")]

    done = asyncio.run(collect())[-1]
    assert calls == []
    assert regular["answer_source"] == "template" and regular["verify"]["ok"]
    assert done["answer_source"] == "template" and done["verify"]["ok"]


@pytest.mark.parametrize("amount,reviewer", [("ผิด", "ผู้ตรวจ"), (-1, "ผู้ตรวจ"),
                                            (None, "ผู้ตรวจ"), (25000, " ")])
def test_admin_cannot_verify_an_invalid_amount_or_anonymous_review(tmp_path, monkeypatch, amount, reviewer):
    row = {"id": "test-only", "amount": 25000, "verified": False}
    source = tmp_path / "facts.json"
    source.write_text(json.dumps([row], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(review, "DATA", tmp_path)

    result = review.verify_fact("test-only", reviewer, amount)

    assert not result["ok"]
    assert json.loads(source.read_text(encoding="utf-8")) == [row]


def test_admin_cannot_approve_ocr_without_a_named_reviewer(tmp_path, monkeypatch):
    row = {"id": "test-chunk", "doc": "ประกาศทดสอบ", "status": "draft", "verified": False}
    source = tmp_path / "chunks_ingested.json"
    source.write_text(json.dumps([row], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(review, "DATA", tmp_path)

    result = review.verify_chunks("ประกาศทดสอบ", " ")

    assert not result["ok"]
    assert json.loads(source.read_text(encoding="utf-8")) == [row]


def test_document_review_approves_both_ocr_and_manual_transcriptions(tmp_path, monkeypatch):
    manual = {"id": "manual", "doc": "ประกาศทดสอบ", "status": "active", "verified": False}
    ocr = {"id": "ocr", "doc": "ประกาศทดสอบ", "status": "draft", "verified": False}
    (tmp_path / "chunks.json").write_text(json.dumps([manual], ensure_ascii=False), encoding="utf-8")
    (tmp_path / "chunks_ingested.json").write_text(json.dumps([ocr], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(review, "DATA", tmp_path)
    monkeypatch.setattr(store, "CHUNKS", [manual.copy(), ocr.copy()])
    monkeypatch.setattr(store, "INGESTED_CHUNKS", [store.CHUNKS[1]])

    result = review.verify_chunks("ประกาศทดสอบ", "ผู้ตรวจ")

    assert result["ok"] and result["approved"] == 2
    for name in ("chunks.json", "chunks_ingested.json"):
        saved = json.loads((tmp_path / name).read_text(encoding="utf-8"))[0]
        assert saved["verified"] is True and saved["reviewed_by"] == "ผู้ตรวจ"


def test_approved_definition_is_available_without_restarting_the_server(tmp_path, monkeypatch):
    term = "ศัพท์ทดสอบใหม่"
    draft = {
        "id": "definition-draft",
        "doc": "ประกาศนิยามทดสอบ",
        "page": 1,
        "clause": "ข้อ 1",
        "fiscal_year": None,
        "status": "draft",
        "verified": False,
        "text": f'“{term}” หมายความว่า ข้อความที่เพิ่งผ่านการตรวจ',
    }
    (tmp_path / "chunks.json").write_text(
        json.dumps([draft], ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(review, "DATA", tmp_path)
    monkeypatch.setattr(store, "CHUNKS", [draft.copy()])
    monkeypatch.setattr(store, "INGESTED_CHUNKS", [])
    monkeypatch.setattr(store, "FACTS", [])
    monkeypatch.setattr(store, "DEFINITIONS", {})
    monkeypatch.setattr(store, "_ANNEX_INDEX", ({}, set()), raising=False)
    monkeypatch.setattr(store, "ANNEX_FUNDS", {})
    monkeypatch.setattr(store, "ANNEX_PAGES", set())

    assert store.define(term) is None
    result = review.verify_chunks("ประกาศนิยามทดสอบ", "ผู้ตรวจ")

    assert result["ok"]
    assert store.define(term)["text"] == "ข้อความที่เพิ่งผ่านการตรวจ"


def test_approved_annex_fact_refreshes_clause_mapping_without_restart(tmp_path, monkeypatch):
    fund = "ทุนทดสอบดัชนี"
    draft = {
        "id": "annex-fact-draft",
        "fund": fund,
        "tier": None,
        "fiscal_year": None,
        "amount_type": "ceiling",
        "amount": 12345,
        "total_cap": None,
        "max_grants_per_person": None,
        "verified": False,
        "source": {
            "doc": "หลักเกณฑ์กลาง",
            "page": 99,
            "clause": "บัญชีแนบท้าย ข้อ 9.9",
        },
    }
    (tmp_path / "facts.json").write_text(
        json.dumps([draft], ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(review, "DATA", tmp_path)
    monkeypatch.setattr(store, "FACTS", [draft.copy()])
    monkeypatch.setattr(store, "CHUNKS", [])
    monkeypatch.setattr(store, "DEFINITIONS", {})
    monkeypatch.setattr(store, "_ANNEX_INDEX", ({}, set()), raising=False)
    monkeypatch.setattr(store, "ANNEX_FUNDS", {})
    monkeypatch.setattr(store, "ANNEX_PAGES", set())
    chunk = {"doc": "หลักเกณฑ์กลาง", "page": 99, "clause": "9.9"}

    assert store.funds_in_clause(chunk) is None
    result = review.verify_fact("annex-fact-draft", "ผู้ตรวจ", 12345)

    assert result["ok"]
    assert store.funds_in_clause(chunk) == [fund]


def test_named_reviewer_can_confirm_a_checked_amount(tmp_path, monkeypatch):
    row = {"id": "test-only", "amount": 25000, "verified": False}
    source = tmp_path / "facts.json"
    source.write_text(json.dumps([row], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(review, "DATA", tmp_path)

    result = review.verify_fact("test-only", "ผู้ตรวจ", 25000)

    saved = json.loads(source.read_text(encoding="utf-8"))[0]
    assert result["ok"] and saved["verified"] is True and saved["reviewed_by"] == "ผู้ตรวจ"


def test_admin_cannot_rewrite_an_already_verified_fact(tmp_path, monkeypatch):
    row = {"id": "trusted-row", "amount": 25000, "verified": True,
           "reviewed_by": "ผู้ตรวจเดิม"}
    source = tmp_path / "facts.json"
    source.write_text(json.dumps([row], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(review, "DATA", tmp_path)
    monkeypatch.setattr(review, "TOKEN", "test-admin-token")

    direct = review.verify_fact("trusted-row", "ผู้ตรวจใหม่", 1)
    response = TestClient(app).post(
        "/v1/admin/verify-fact",
        headers={"X-Admin-Token": "test-admin-token"},
        json={"id": "trusted-row", "reviewer": "ผู้ตรวจใหม่", "amount": 1},
    )

    assert not direct["ok"]
    assert response.status_code == 409
    assert json.loads(source.read_text(encoding="utf-8")) == [row]


def test_admin_can_open_only_a_known_source_pdf_with_a_token(tmp_path, monkeypatch):
    original = b"%PDF-1.4\nsource document\n%%EOF"
    (tmp_path / "ประกาศ_42-68_ทุนส่งเสริมวิจัย_ปีงบประมาณ_พ.ศ._2569.pdf").write_bytes(original)
    (tmp_path / "secret.pdf").write_bytes(b"not a source")
    monkeypatch.setattr(review, "SOURCE_DIR", tmp_path)
    monkeypatch.setattr(review, "TOKEN", "test-admin-token")
    client = TestClient(app)

    unauthorised = client.get("/v1/admin/source", params={"doc": "ประกาศ 42/2568"})
    unknown = client.get("/v1/admin/source", params={"doc": "../secret"},
                         headers={"X-Admin-Token": "test-admin-token"})
    source = client.get("/v1/admin/source", params={"doc": "ประกาศ 42/2568"},
                        headers={"X-Admin-Token": "test-admin-token"})

    assert unauthorised.status_code == 401
    assert unknown.status_code == 404
    assert source.status_code == 200 and source.content == original
    assert source.headers["content-type"].startswith("application/pdf")
    assert source.headers["cache-control"] == "no-store, private"


def test_admin_pending_includes_every_unreviewed_ocr_chunk(monkeypatch):
    chunks = [{"id": f"test-{i}", "doc": "ประกาศทดสอบ", "status": "draft",
               "verified": False, "page": i // 20 + 1, "clause": "ข้อ 1", "text": "ข้อความ OCR"}
              for i in range(205)]
    monkeypatch.setattr(store, "CHUNKS", chunks)
    monkeypatch.setattr(store, "FACTS", [])

    pending = review.pending()

    assert pending["chunk_count"] == 205
    assert len(pending["chunks"]) == 205


def test_admin_pending_includes_rows_with_a_missing_verified_marker(monkeypatch):
    fact = {"id": "missing-marker-fact"}
    chunk = {"id": "missing-marker-chunk", "doc": "ประกาศทดสอบ"}
    monkeypatch.setattr(store, "FACTS", [fact])
    monkeypatch.setattr(store, "CHUNKS", [chunk])

    pending = review.pending()

    assert pending["facts"] == [fact]
    assert pending["chunks"] == [chunk]


def test_admin_page_offers_original_pdf_and_ocr_review():
    page = TestClient(app).get("/admin")

    assert page.status_code == 200
    assert "เปิด PDF ต้นฉบับ" in page.text
    assert "ดูข้อความ OCR" in page.text
    assert "ตรวจ PDF ใหม่ก่อนนำเข้าระบบ" in page.text
    assert "ไม่บันทึกไฟล์" in page.text


def _sample_text_pdf() -> bytes:
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((40, 60), "Grant announcement text for inspection.\n" * 8)
    data = document.tobytes()
    document.close()
    return data


def test_pdf_inspection_reads_text_but_never_marks_it_verified():
    from app import pdf_inspection

    report = pdf_inspection.inspect(_sample_text_pdf(), "new-fund.pdf")

    assert report["pages_total"] == 1
    assert report["pages"][0]["method"] == "text"
    assert "Grant announcement" in report["pages"][0]["text"]
    assert report["pages"][0]["status"] == "draft"
    assert report["pages"][0]["verified"] is False
    assert report["saved"] is False


def test_pdf_inspection_marks_ocr_numbers_as_unverified_candidates(monkeypatch):
    from app import pdf_inspection

    monkeypatch.setattr(pdf_inspection, "page_text", lambda _page, _dpi: ("วงเงิน 100,000 บาท", "ocr"))
    report = pdf_inspection.inspect(_sample_text_pdf(), "scan.pdf")

    assert report["pages"][0]["method"] == "ocr"
    assert report["pages"][0]["amount_candidates"] == ["100,000 บาท"]
    assert report["pages"][0]["verified"] is False


def test_pdf_inspection_explains_when_a_scanned_page_cannot_be_read(monkeypatch):
    from app import pdf_inspection

    monkeypatch.setattr(pdf_inspection, "page_text", lambda _page, _dpi: ("", "needs_ocr"))
    report = pdf_inspection.inspect(_sample_text_pdf(), "scan.pdf")

    assert report["pages"][0]["text"] == ""
    assert "OCR" in report["note"]
    assert report["pages"][0]["verified"] is False


def test_short_pdf_text_is_not_lost_when_ocr_is_unavailable(monkeypatch):
    from scripts import ingest

    class ShortTextPage:
        def get_text(self):
            return "Short grant notice"

    monkeypatch.setattr(ingest, "_ocr", lambda _page, _dpi: ("", "needs_ocr"))
    assert ingest.page_text(ShortTextPage(), 200) == ("Short grant notice", "text_partial")


def test_pdf_inspection_api_requires_admin_token_and_rejects_non_pdfs(monkeypatch):
    monkeypatch.setattr(review, "TOKEN", "test-admin-token")
    client = TestClient(app)
    headers = {"X-Admin-Token": "test-admin-token", "X-Pdf-Name": "new-fund.pdf",
               "Content-Type": "application/pdf"}

    assert client.post("/v1/admin/inspect-pdf", content=_sample_text_pdf()).status_code == 401
    invalid = client.post("/v1/admin/inspect-pdf", content=b"not a pdf", headers=headers)
    valid = client.post("/v1/admin/inspect-pdf", content=_sample_text_pdf(), headers=headers)

    assert invalid.status_code == 400
    assert valid.status_code == 200
    assert valid.json()["pages"][0]["verified"] is False


def test_pdf_inspection_api_rejects_oversized_upload_before_parsing(monkeypatch):
    from app import pdf_inspection

    monkeypatch.setattr(review, "TOKEN", "test-admin-token")
    monkeypatch.setattr(pdf_inspection, "MAX_BYTES", 16)
    client = TestClient(app)
    response = client.post("/v1/admin/inspect-pdf", content=_sample_text_pdf(),
                           headers={"X-Admin-Token": "test-admin-token", "X-Pdf-Name": "large.pdf",
                                    "Content-Type": "application/pdf"})

    assert response.status_code == 413


def test_cli_ingest_never_auto_approves_a_text_pdf(tmp_path, monkeypatch):
    from scripts import ingest

    (tmp_path / "new-fund.pdf").write_bytes(_sample_text_pdf())
    output = tmp_path / "chunks.json"
    monkeypatch.setattr(sys, "argv", ["ingest.py", "--src", str(tmp_path), "--out", str(output)])

    assert ingest.main() == 0
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert rows and all(row["status"] == "draft" and row["verified"] is False for row in rows)


def test_demo_home_starts_with_reviewed_examples_and_explains_pending_amounts():
    page = TestClient(app).get("/")

    assert page.status_code == 200
    assert "เพดานตามหลักเกณฑ์กลางของทุนวิจัยสถาบัน" in page.text
    assert "เงินสมทบหมายถึงอะไร" in page.text
    assert "ทุนวิจัยสถาบันมีรายละเอียดอะไรบ้าง" in page.text
    assert "แนะนำทุนสำหรับตีพิมพ์บทความ" not in page.text
    assert "วงเงินจัดสรรรายปีจากเอกสารสแกนยังรอเจ้าหน้าที่ตรวจ" in page.text
    assert "หลักเกณฑ์กลางที่มีอยู่ยังไม่ระบุเลขที่และวันประกาศ" in page.text


def test_audit_log_records_failed_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LOG", tmp_path / "queries.jsonl")
    audit.record({"question": "ทดสอบ", "verify": {"ok": False}, "ruling": {}}, "P1")

    row = json.loads((tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["blocked"] is True


def test_the_audit_log_keeps_the_question_and_not_the_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LOG", tmp_path / "queries.jsonl")
    audit.record({"question": "ทุนวิจัยสถาบันได้เท่าไร", "answer": "25,000 บาท",
                  "slots": {"fund": "ทุนวิจัยสถาบัน"}, "ruling": {"verdict": "eligible"},
                  "citations": ["ประกาศ 42/2568"], "answer_source": "template",
                  "timings_ms": {"total": 12.0}}, "P1")
    rows = [json.loads(line) for line in (tmp_path / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["user_id"] == "P1" and rows[0]["verdict"] == "eligible"
    assert "staff_type" not in json.dumps(rows[0], ensure_ascii=False)


def test_audit_tail_skips_a_partial_or_malformed_line(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LOG", tmp_path / "queries.jsonl")
    audit.LOG.write_text(
        '{"question":"เก่า"}\n{broken\n{"question":"ล่าสุด"}\n',
        encoding="utf-8",
    )

    assert [row["question"] for row in audit.tail(20)] == ["เก่า", "ล่าสุด"]


def test_audit_tail_bounds_the_requested_result_size(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LOG", tmp_path / "queries.jsonl")
    audit.LOG.write_text('{"n":1}\n{"n":2}\n', encoding="utf-8")

    assert audit.tail(0) == []
    assert audit.tail(10_000) == [{"n": 1}, {"n": 2}]


def test_audit_log_rotates_before_it_can_grow_without_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LOG", tmp_path / "queries.jsonl")
    monkeypatch.setattr(audit, "MAX_LOG_BYTES", 1, raising=False)
    monkeypatch.setattr(audit, "MAX_BACKUPS", 1, raising=False)

    audit.record({"question": "รายการแรก", "answer": "คำตอบ", "ruling": {}}, "P1")
    audit.record({"question": "รายการล่าสุด", "answer": "คำตอบ", "ruling": {}}, "P1")

    backup = tmp_path / "queries.jsonl.1"
    assert backup.is_file()
    assert "รายการแรก" in backup.read_text(encoding="utf-8")
    assert "รายการล่าสุด" in audit.LOG.read_text(encoding="utf-8")
    assert [row["question"] for row in audit.tail(2)] == ["รายการแรก", "รายการล่าสุด"]


def test_audit_tail_reads_only_a_bounded_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "LOG", tmp_path / "queries.jsonl")
    monkeypatch.setattr(audit, "MAX_TAIL_READ_BYTES", 64, raising=False)
    audit.LOG.write_text(
        "".join(json.dumps({"n": n}) + "\n" for n in range(100)),
        encoding="utf-8",
    )

    def whole_file_read_is_forbidden(*_args, **_kwargs):
        raise AssertionError("tail loaded the whole audit log")

    monkeypatch.setattr(Path, "read_text", whole_file_read_is_forbidden)

    rows = audit.tail(3)

    assert rows == [{"n": 97}, {"n": 98}, {"n": 99}]
