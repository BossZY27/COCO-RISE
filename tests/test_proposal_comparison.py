"""Guard the proposal comparison against using unreviewed evidence."""

import asyncio

from app.infrastructure.ollama import legacy as llm
from app.modules.knowledge import corpus as store
from app.modules.query import engine as pipeline
from scripts.compare_proposal_retrieval import load_cases, rank_methods, score, verified_fact_coverage


def test_comparison_uses_only_approved_gold_evidence():
    cases = load_cases()
    approved = {chunk["id"] for chunk in store.CHUNKS if store._passes_filter(chunk, None)}
    assert len(cases) == 78
    assert {case["evidence_id"] for case in cases} <= approved


def test_all_comparison_methods_exclude_draft_chunks(monkeypatch):
    approved = next(chunk for chunk in store.CHUNKS if store._passes_filter(chunk, None))
    draft = next(chunk for chunk in store.CHUNKS if not store._passes_filter(chunk, None))
    monkeypatch.setattr(store, "_VECTORS", {approved["id"]: [1.0, 0.0], draft["id"]: [1.0, 0.0]})
    results = rank_methods(approved["text"], [1.0, 0.0])
    for hits in results.values():
        assert all(store._passes_filter(chunk, None) for chunk in hits)
        assert draft["id"] not in {chunk["id"] for chunk in hits}


def test_structured_fact_coverage_requires_a_matching_approved_source():
    cases = load_cases()
    covered, total = verified_fact_coverage(cases)
    assert total == 34
    assert covered <= total
    assert covered > 0
    assert score([1, 4, None]) == {"total": 3, "hit1": 1, "hit3": 1, "hit5": 2, "mrr5": 0.417}


def test_weighted_fusion_can_prefer_semantic_evidence_without_dropping_other_paths():
    keyword = [{"id": "word"}]
    semantic = [{"id": "meaning"}]
    hierarchy = [{"id": "parent"}]

    ranked = store.rrf([keyword, semantic, hierarchy], weights=(1.0, 2.0, 1.0))

    assert [hit["id"] for hit in ranked] == ["meaning", "word", "parent"]


def test_guarded_fusion_keeps_vector_top_three_and_adds_other_evidence():
    semantic = [{"id": str(index)} for index in range(1, 6)]
    keyword = [{"id": "keyword"}, {"id": "1"}]
    hierarchy = [{"id": "hierarchy"}, {"id": "1"}]

    ranked = store.semantic_guarded_rrf(keyword, semantic, hierarchy)

    assert [hit["id"] for hit in ranked[:3]] == ["1", "2", "3"]
    assert "keyword" in {hit["id"] for hit in ranked[3:]}
    assert "hierarchy" in {hit["id"] for hit in ranked[3:]}


def test_pipeline_uses_guarded_fusion_only_for_real_vector_results(monkeypatch):
    semantic = [{"id": str(index)} for index in range(1, 6)]
    keyword = [{"id": "keyword"}, {"id": "1"}]
    hierarchy = [{"id": "hierarchy"}, {"id": "1"}]
    monkeypatch.setattr(store, "keyword_search", lambda *_args: keyword)
    monkeypatch.setattr(store, "hierarchical_search", lambda *_args: hierarchy)
    monkeypatch.setattr(store, "fact_lookup", lambda *_args: [])
    slots = {"standalone_question": "เงื่อนไขทุนวิจัยสถาบัน", "fiscal_year": 2569,
             "intent": "definition", "fund": "ทุนวิจัยสถาบัน", "tier": None}

    monkeypatch.setattr(pipeline, "_semantic", lambda *_args: (semantic, llm.EMBED_MODEL))
    vector_result = asyncio.run(pipeline.retrieve(slots))
    assert [row["id"] for row in vector_result["chunks"][:3]] == ["1", "2", "3"]

    monkeypatch.setattr(pipeline, "_semantic", lambda *_args: (semantic, "trigram"))
    fallback_result = asyncio.run(pipeline.retrieve(slots))
    assert fallback_result["chunks"] == store.rrf([keyword, semantic, hierarchy])
