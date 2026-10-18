"""Tests for reading questions the way people type them, and for answers built from several rows.

Like tests/test_core.py, nothing here writes to data/. The model is switched off for every test, so each
answer checked is the template the verify gate sees, and the suite never waits on Ollama.

    python -m pytest tests -q
"""
import asyncio
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import answers, llm, pipeline, store, understand  # noqa: E402

ANNEX = "หลักเกณฑ์กลาง"


@pytest.fixture(autouse=True)
def template_only(monkeypatch):
    monkeypatch.setattr(llm, "MODE", "template")
    monkeypatch.setattr(store, "vectors_ready", lambda: False)


def ask(question: str, persona: str = "P1", history: list[str] | None = None) -> dict:
    return asyncio.run(pipeline.run(question, persona, history))


def annex_chunk(page: int, clause: str) -> dict:
    return next(c for c in store.CHUNKS
                if c.get("doc") == ANNEX and c.get("page") == page and c.get("clause") == clause)


# ---------------------------------------------------------------- how people type
def test_spelling_variants_are_read_as_one_word():
    assert understand.normalize("ทุนวิจัยสถาบันปิดรับเมื่อไหร่") == "ทุนวิจัยสถาบันปิดรับเมื่อไร"
    assert understand.normalize("ค่าตอบเเทนได้มั้ย") == "ค่าตอบแทนได้ไหม"
    assert understand.normalize("ปีงบ  ๒๕๖๙") == "ปีงบ 2569"


def test_a_closing_date_question_is_not_answered_with_money():
    slots = pipeline.analyze("ทุนวิจัยสถาบันปิดรับเมื่อไหร่")
    assert slots["intent"] == "application" and slots["fund"] == "ทุนวิจัยสถาบัน"


def test_a_money_figure_is_not_read_as_a_fiscal_year():
    # "25000" holds "2500", which the year pattern once took for fiscal year 2500
    slots = pipeline.analyze("ทุนสถาบัน 25000 ใช่ไหม")
    assert slots["fiscal_year"] == pipeline.DEFAULT_YEAR and slots["year_assumed"]
    assert pipeline.analyze("ทุนวิจัยสถาบันปี 2570 ได้เท่าไร")["fiscal_year"] == 2570
    assert understand.relative_year("ขอได้ปี 60,000 บาท", 2569) is None


@pytest.mark.parametrize("ending", ["ได้ไหม", "ได้มั้ย", "ได้มั๊ย", "ได้หรือไม่", "ได้หรือเปล่า", "ได้รึเปล่า",
                                    "ได้ป่าว", "ได้ปะ", "ได้บ่"])
def test_every_way_of_asking_yes_or_no_is_an_eligibility_question(ending):
    assert pipeline.analyze(f"คนนอกขอทุน{ending}")["intent"] == "eligibility"


def test_how_often_is_not_a_yes_or_no_ending():
    assert not understand.asks_if_possible("ขอทุนได้บ่อยแค่ไหน")


def test_an_outsider_question_ending_in_rue_plao_gets_the_condition():
    result = ask("คนนอกขอทุนได้หรือเปล่า")
    assert "ผู้ขอรับทุนคือบุคลากร" in result["answer"] and result["verify"]["ok"]


# ---------------------------------------------------------------- which fund a clause sets
def test_an_annex_clause_maps_to_the_fund_its_reviewed_row_cites():
    # Five funds begin "ค่าตอบแทนผลงาน", so matching the first letters of a name tied clause 2.5 to all
    # five, and a suggestion list cut at four dropped the one the clause is about.
    assert store.funds_mentioned_in([annex_chunk(6, "2.5")]) == ["ค่าตอบแทนผลงานที่ได้รับการอ้างอิงสูง"]
    assert set(store.funds_mentioned_in([annex_chunk(7, "2.11")])) == {
        "ทุนนำเสนอผลงานทางวิชาการ", "ทุนนำเสนอผลงานนวัตกรรมและสิ่งประดิษฐ์"}
    assert store.funds_mentioned_in([annex_chunk(5, "2.1.1")]) == ["ทุนทำวิจัยในต่างประเทศ"]


def test_a_clause_no_reviewed_row_covers_suggests_no_fund():
    # 3.1 is a group 3 fund the system does not hold. Its wording once pointed at two group 1 funds.
    assert store.funds_mentioned_in([annex_chunk(8, "3.1")]) == []


def test_a_purpose_without_a_fund_name_does_not_trigger_matching():
    text, citations, _ = answers.funds_for_purpose("ผลงานที่มีคนอ้างอิงเยอะ", [])
    assert "ยังไม่เปิดระบบแนะนำทุน" in text
    assert "ค่าตอบแทนผลงานที่ได้รับการอ้างอิงสูง" not in text
    assert not citations


def test_a_citation_count_question_finds_its_fund():
    assert understand.mentions("บทความที่ได้รับการอ้างอิงเยอะได้เงินไหม")[0]["fund"] == \
        "ค่าตอบแทนผลงานที่ได้รับการอ้างอิงสูง"
    # อ้างอิง alone also means citing a source, so on its own it names no fund
    assert understand.mentions("ข้อมูลนี้อ้างอิงจากเอกสารไหน") == []


# ---------------------------------------------------------------- funds by subject
def test_a_subject_question_is_read_as_a_subject():
    assert understand.topic_of("ทุนเกี่ยวกับการสอนมีอะไรบ้าง") == "การสอน"
    assert understand.topic_of("มีทุนอะไรบ้างที่เกี่ยวข้องกับการตีพิมพ์") == "การตีพิมพ์"
    assert understand.topic_of("ทุนด้านนวัตกรรมมีไหม") == "นวัตกรรม"
    assert understand.topic_of("ทุนวิจัยสถาบันได้เท่าไร") == ""


def test_a_subject_lists_every_fund_whose_clause_names_it():
    text, citations, used = answers.funds_about("การสอน", [])
    assert "ค่าตอบแทนสื่อการสอนออนไลน์" in text
    assert "ทุนสนับสนุนงานวิจัย นวัตกรรม และสิ่งประดิษฐ์เพื่อการเรียนการสอน" in text
    assert citations and used == []


def test_a_subject_most_clauses_mention_does_not_list_them_all():
    text, _, _ = answers.funds_about("วิจัย", [])
    assert "มีทุนอะไรบ้าง" in text and text.count(",") < answers.TOPIC_LIST_MAX


def test_a_subject_no_clause_names_says_so():
    text, _, _ = answers.funds_about("การท่องเที่ยว", [])
    assert "ไม่พบทุน" in text


def test_a_subject_question_is_answered_end_to_end():
    result = ask("มีทุนเกี่ยวกับการตีพิมพ์บ้างไหม")
    assert result["slots"]["intent"] == "topic"
    assert "เงินสมทบค่าธรรมเนียมตีพิมพ์" in result["answer"] and result["verify"]["ok"]


# ---------------------------------------------------------------- who is asking, and what for
def test_a_generic_word_is_not_a_purpose():
    assert understand.purpose_of("ขอทุนวิจัยได้ไหม") == ""
    assert understand.purpose_of("ขอทุนทำวิจัยได้หรือเปล่า") == ""


def test_words_before_the_grant_describe_the_person_not_the_purpose():
    assert understand.purpose_of("นักศึกษาปริญญาเอกขอทุนได้ไหม") == ""
    assert understand.purpose_of("ขอทุนซื้อรถได้ไหม") == "ซื้อรถ"
    assert "ญี่ปุ่น" in understand.purpose_of("อยากได้ทุนไปญี่ปุ่น")


def test_a_leisure_trip_is_not_rewritten_as_research_abroad():
    question = "อยากได้ทุนไปเที่ยวญี่ปุ่น"

    assert understand.with_places_abroad(question) == question


def test_a_question_without_a_subject_is_about_the_asker():
    inside = store.PERSONAS["P1"]
    answer, _ = answers.who_may_apply("ขอทุนวิจัยได้ไหม", inside)
    assert answer.startswith("ตามข้อมูลที่ระบบมี คุณอยู่ในกลุ่มผู้ขอรับทุน")
    general, _ = answers.who_may_apply("ใครขอทุนได้บ้าง", inside)
    assert "คุณอยู่ในกลุ่ม" not in general


def test_a_student_question_gets_the_definition_not_a_postdoc_fund():
    result = ask("นักศึกษาปริญญาเอกขอทุนได้ไหม")
    assert "ผู้ขอรับทุนคือบุคลากร" in result["answer"]
    assert "หลังปริญญาเอก" not in result["answer"]


@pytest.mark.parametrize("persona", ["P1", "P2", "P3", "P4", "P6"])
def test_answers_about_the_asker_never_repeat_their_record(persona):
    # Answer text is logged. Employment type, affiliation and name are personal data under PDPA.
    profile = store.PERSONAS[persona]
    for question in ("ผมขอทุนได้ไหม", "ขอทุนได้ไหม", "ฉันมีสิทธิ์ขอทุนไหม"):
        answer, _ = answers.who_may_apply(question, profile)
        assert profile["name"] not in answer
        if profile["department"] != answers.COLLEGE:
            assert profile["department"] not in answer
    _, sentence = answers.applicant_status(profile)
    assert profile["staff_type"] not in sentence and profile["department"] not in sentence


# ---------------------------------------------------------------- answers built from several rows
@pytest.mark.parametrize("persona, question, figures", [
    ("P1", "ทุนนักวิจัยอาวุโสกับทุนวิจัยในชั้นเรียนต่างกันยังไง", ["1,000,000", "10,000"]),
    ("P1", "ทุนอะไรให้เงินต่ำที่สุด", ["10,000"]),
    ("P1", "ทุนนวัตกรรมความเป็นเลิศแต่ละระดับได้เท่าไร", ["100,000", "200,000", "250,000"]),
    ("P1", "ค่าตอบแทนสื่อการสอนออนไลน์มีเงื่อนไขอะไร", ["40,000", "240,000", "ข้อ 8"]),
    ("P6", "ถ้าได้เงินรางวัลผู้แต่งชื่อแรกแล้ว จะขอเงินรางวัลผู้ประพันธ์บรรณกิจได้หรือไม่", ["ไม่ได้"]),
])
def test_answers_built_from_several_rows_pass_the_gate(persona, question, figures):
    result = ask(question, persona)
    assert result["answer_source"] == "template" and result["verify"]["ok"], result["verify"]
    assert all(figure in result["answer"] for figure in figures), result["answer"]


def test_a_tiered_fund_is_never_described_as_untiered():
    tiered = {f["fund"] for f in store.FACTS if f["tier"] and f.get("verified")}
    assert tiered
    for fund in tiered:
        text, rows = answers.fund_summary(fund)
        assert "แยกตามชั้น" in text and "ไม่ได้แยกตามชั้น" not in text
        assert any(r["tier"] for r in rows)


def test_the_pending_notice_keeps_count_cap_and_condition():
    text, citations, _ = pipeline._pending_notice(
        {"fund": "ทุนสนับสนุนการได้รับทุนวิจัยภายนอก", "tier": None, "fiscal_year": 2569})
    assert "100,000" in text and "300,000" in text and "ร้อยละ 10" in text and citations
    text, _, _ = pipeline._pending_notice({"fund": "ทุนวิจัยสถาบัน", "tier": None, "fiscal_year": 2569})
    assert "2 ทุนต่อปี" in text and "50,000" in text


def test_every_figure_in_a_reviewed_note_is_in_the_clause_it_came_from():
    # Notes are typed by hand beside the clause they summarise. A figure the clause does not hold would
    # be a reviewed-looking number nobody can find in the document.
    for row in store.FACTS:
        note = row.get("note") or ""
        if not row.get("verified") or row["source"]["doc"] != ANNEX or not re.search(r"\d", note):
            continue
        clause = row["source"]["clause"].replace("บัญชีแนบท้าย ข้อ", "").strip()
        text = "".join(c["text"] for c in store.CHUNKS if c.get("doc") == ANNEX and
                       (c.get("clause") == clause or str(c.get("clause", "")).startswith(clause + ".")))
        flat = text.replace(",", "")
        for number in re.findall(r"\d[\d,]*", note):
            assert number.replace(",", "") in flat, (row["id"], number)


# ---------------------------------------------------------------- names that mean more than one fund
def test_every_alias_alone_names_its_fund():
    for alias, fund in understand.ALIASES:
        assert understand.mentions(alias)[0]["fund"] == fund, alias


def test_an_ambiguous_name_takes_its_hint_or_the_first_fund():
    plain = understand.mentions("ทุนความเป็นเลิศ")[0]
    assert plain["fund"] == "ทุนวิจัยความเป็นเลิศ" and plain["alternatives"] == ["ทุนนวัตกรรมความเป็นเลิศ"]
    hinted = understand.mentions("ทุนความเป็นเลิศด้านนวัตกรรม")[0]
    assert hinted["fund"] == "ทุนนวัตกรรมความเป็นเลิศ" and hinted["alternatives"] == []


def test_an_ambiguous_name_says_which_fund_it_answered():
    result = ask("ทุนความเป็นเลิศได้เท่าไร")
    assert result["slots"]["fund"] == "ทุนวิจัยความเป็นเลิศ"
    assert "ทุนนวัตกรรมความเป็นเลิศ" in result["answer"] and result["verify"]["ok"], result["verify"]


def test_a_tier_asked_of_an_untiered_fund_is_pointed_out():
    result = ask("ทุนวิจัยสถาบันชั้นสูงได้เท่าไร")
    assert "ไม่ได้แบ่งเป็นชั้น" in result["answer"] and result["verify"]["ok"], result["verify"]
    assert "ไม่ได้แบ่งเป็นชั้น" not in ask("ทุนวิจัยความเป็นเลิศชั้นสูงได้เท่าไร")["answer"]


@pytest.mark.parametrize("text, year", [("ปีหน้า", 2570), ("ปีที่แล้ว", 2568), ("ปีงบ 69", 2569),
                                        ("ปี พ.ศ. 70", 2570), ("ปีนี้", 2569), ("ทุนนี้", None)])
def test_relative_and_two_digit_years(text, year):
    assert understand.relative_year(text, 2569) == year


# ---------------------------------------------------------------- small talk
@pytest.mark.parametrize("question, kind", [("สวัสดีครับ", "greeting"), ("หวัดดีค่ะ", "greeting"),
                                            ("ขอบคุณมากครับ", "thanks"), ("ระบบนี้ทำอะไรได้บ้าง", "capability")])
def test_small_talk_is_recognised(question, kind):
    assert understand.small_talk(question) == kind


def test_a_grant_question_with_a_greeting_is_still_a_grant_question():
    assert understand.small_talk("สวัสดีครับ ทุนวิจัยสถาบันได้เท่าไร") is None
    assert understand.small_talk("highly cited ได้เท่าไร") is None


def test_small_talk_gets_a_reply_instead_of_a_fund_guess():
    result = ask("สวัสดีครับ")
    assert result["slots"]["intent"] == "smalltalk" and result["verify"]["ok"]
    assert "ไม่แน่ใจว่าคำถามนี้หมายถึงทุนใด" not in result["answer"] and "บาท" not in result["answer"]
