import json
from pathlib import Path

from app.modules.knowledge import corpus as store
from evaluation.retrieval import RetrievalComparison


def _chunk(chunk_id, doc, page, text, verified=True, clause="ทั้งหน้า", fiscal_year=None):
    return {"id": chunk_id, "doc": doc, "page": page, "text": text,
            "status": "active", "verified": verified, "fiscal_year": fiscal_year, "clause": clause}


def test_both_methods_exclude_unreviewed_chunks():
    index = RetrievalComparison([
        _chunk("approved", "criteria", 1, "ทุนวิจัยในชั้นเรียน"),
        _chunk("draft", "scan", 2, "ทุนวิจัยในชั้นเรียน", verified=False),
    ])
    for method in (index.flat, index.page_only, index.parent_context, index.hierarchical):
        assert [row["id"] for row in method("ทุนวิจัยในชั้นเรียน")] == ["approved"]


def test_hierarchy_only_returns_children_from_selected_pages():
    index = RetrievalComparison([
        _chunk("target", "criteria", 1, "ทุนวิจัยในชั้นเรียน"),
        _chunk("other", "criteria", 2, "ค่าตอบแทนสื่อการสอนออนไลน์"),
    ])
    assert [row["id"] for row in index.hierarchical("ทุนวิจัยในชั้นเรียน", top_pages=1)] == ["target"]


def test_hierarchy_inherits_only_approved_parent_heading():
    index = RetrievalComparison([
        _chunk("parent", "criteria", 1, "1.6 ทุนวิจัยความเป็นเลิศ วงเงินรวม", clause="1.6"),
        _chunk("child", "criteria", 1, "1.6.1 ระดับต้น ไม่เกิน", clause="1.6.1"),
        _chunk("draft_parent", "scan", 2, "1.6 ทุนลับ วงเงินรวม", verified=False, clause="1.6"),
        _chunk("orphan", "scan", 2, "1.6.1 ระดับต้น ไม่เกิน", clause="1.6.1"),
    ])
    assert "ทุนวิจัยความเป็นเลิศ" in index.hier_tokens[1]
    assert "ทุนลับ" not in index.hier_tokens[2]


def test_all_methods_filter_out_other_fiscal_years():
    index = RetrievalComparison([
        _chunk("current", "announcement", 1, "ทุนวิจัยสถาบัน", fiscal_year=2569),
        _chunk("other_year", "announcement", 1, "ทุนวิจัยสถาบัน", fiscal_year=2570),
    ], fiscal_year=2569)
    for method in (index.flat, index.page_only, index.parent_context, index.hierarchical):
        assert [row["id"] for row in method("ทุนวิจัยสถาบัน")] == ["current"]


def test_unstructured_page_label_does_not_inherit_another_pages_text():
    index = RetrievalComparison([
        _chunk("first", "criteria", 1, "ประกาศมหาวิทยาลัยขอนแก่น"),
        _chunk("second", "criteria", 2, "เงินสมทบหมายความว่าเงินที่ให้โดยไม่ต้องชดใช้คืน"),
    ])
    assert "ประกาศมหาวิทยาลัยขอนแก่น" not in index.hier_tokens[1]


def test_uploaded_documents_with_same_label_do_not_share_parent_or_page():
    parent = _chunk("old_parent", "ประกาศทุน", 1, "ข้อ 4 ทุนวิจัยสถาบัน", clause="ข้อ 4")
    parent["document_id"] = "doc_old"
    child = _chunk("new_child", "ประกาศทุน", 1, "ข้อ 4 (1.1) วงเงิน", clause="ข้อ 4 (1.1)")
    child.update(document_id="doc_new", parent_clause="ข้อ 4")
    index = RetrievalComparison([parent, child])
    assert len(index.page_children) == 2
    assert "ทุนวิจัยสถาบัน" not in index.hier_tokens[1]


def test_explicit_parent_clause_from_pdf_chunking_is_used_only_after_approval():
    parent = _chunk("parent", "ประกาศทุน", 1, "ข้อ 4 ทุนวิจัยในชั้นเรียน", clause="ข้อ 4")
    child = _chunk("child", "ประกาศทุน", 1, "ข้อ 4 (1.1) ไม่เกิน", clause="ข้อ 4 (1.1)")
    child["parent_clause"] = "ข้อ 4"
    draft = _chunk("draft", "ประกาศทุน", 1, "ข้อ 4 (1.2) วงเงิน", verified=False,
                   clause="ข้อ 4 (1.2)")
    draft["parent_clause"] = "ข้อ 4"
    index = RetrievalComparison([parent, child, draft])
    assert "ทุนวิจัยในชั้นเรียน" in index.hier_tokens[1]
    assert "draft" not in [row["id"] for row in index.hierarchical("ทุนวิจัยในชั้นเรียน")]


def test_live_hierarchy_reads_fresh_approved_snapshot_without_draft(monkeypatch):
    approved = _chunk("new_approved", "ประกาศใหม่", 1, "ทุนทำวิจัยมหาสมุทร", fiscal_year=2570)
    draft = _chunk("new_draft", "ประกาศใหม่", 1, "ทุนทำวิจัยมหาสมุทร", verified=False,
                   fiscal_year=2570)
    monkeypatch.setattr(store, "CHUNKS", [approved, draft])
    assert [row["id"] for row in store.hierarchical_search("ทุนทำวิจัยมหาสมุทร", 2570)] == ["new_approved"]
    assert store.hierarchical_search("ทุนทำวิจัยมหาสมุทร", 2569) == []


def test_human_review_refresh_makes_new_chunk_searchable(monkeypatch):
    draft = _chunk("awaiting_person", "ประกาศใหม่", 1, "ทุนทำวิจัยมหาสมุทร", verified=False)
    original_chunks = store.CHUNKS
    try:
        monkeypatch.setattr(store, "CHUNKS", [draft])
        assert store.hierarchical_search("ทุนทำวิจัยมหาสมุทร", None) == []
        draft.update(verified=True)
        store.refresh_derived_indexes()
        assert [row["id"] for row in store.hierarchical_search("ทุนทำวิจัยมหาสมุทร", None)] == ["awaiting_person"]
    finally:
        monkeypatch.setattr(store, "CHUNKS", original_chunks)
        store.refresh_derived_indexes()


def test_benchmark_gold_is_approved_and_unique():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    cases = []
    for name in ("retrieval_benchmark.json", "retrieval_benchmark_holdout.json"):
        cases.extend(json.loads((data_dir / name).read_text(encoding="utf-8")))
    approved = {row["id"] for row in store.CHUNKS if store._passes_filter(row, None)}
    assert len(cases) == len({case["question"] for case in cases})
    assert {case["evidence_id"] for case in cases} <= approved
