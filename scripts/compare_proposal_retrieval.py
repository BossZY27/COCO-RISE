"""Compare the three retrieval stages in the proposal on approved evidence.

This is an offline retrieval experiment. It does not generate answers or approve
new data. A real BGE-M3 embedding is required; it never substitutes trigrams.
"""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import llm, pipeline, store  # noqa: E402
from app.retrieval_experiment import RetrievalComparison  # noqa: E402
from scripts.compare_retrieval import systematic_cases  # noqa: E402

METHODS = ("semantic", "hybrid_text", "proposed_retrieval", "semantic_weighted_proposed",
           "semantic_guarded_proposed")


def load_cases() -> list[dict]:
    index = RetrievalComparison(store.CHUNKS)
    suites = (
        ("initial", json.loads((ROOT / "data/retrieval_benchmark.json").read_text(encoding="utf-8"))),
        ("reviewed_holdout", json.loads((ROOT / "data/retrieval_benchmark_holdout.json").read_text(encoding="utf-8"))),
        ("systematic", systematic_cases(index)),
    )
    cases = [{**case, "suite": suite} for suite, rows in suites for case in rows]
    approved = {row["id"] for row in index.chunks}
    if len({case["question"] for case in cases}) != len(cases):
        raise ValueError("Benchmark questions are not unique")
    if any(case["evidence_id"] not in approved for case in cases):
        raise ValueError("Gold evidence is not approved")
    return cases


def rank_methods(question: str, vector: list[float], top_k: int = 5) -> dict[str, list[dict]]:
    semantic = store.vector_search(vector, None, top_k=10)
    keyword = store.keyword_search(question, None, top_k=10)
    hierarchy = store.hierarchical_search(question, None, top_k=10)
    fused = store.rrf([keyword, semantic, hierarchy], top_k=10)
    return {
        "semantic": semantic[:top_k],
        "hybrid_text": store.rrf([keyword, semantic], top_k=top_k),
        "proposed_retrieval": fused[:top_k],
        "semantic_weighted_proposed": store.rrf(
            [keyword, semantic, hierarchy], top_k=top_k, weights=(1.0, 2.0, 1.0)),
        "semantic_guarded_proposed": store.semantic_guarded_rrf(
            keyword, semantic, hierarchy, top_k=top_k),
    }


def score(ranks: list[int | None]) -> dict[str, float | int]:
    total = len(ranks)
    return {
        "total": total,
        "hit1": sum(rank == 1 for rank in ranks),
        "hit3": sum(rank is not None and rank <= 3 for rank in ranks),
        "hit5": sum(rank is not None and rank <= 5 for rank in ranks),
        "mrr5": round(statistics.mean(1 / rank if rank else 0 for rank in ranks), 3),
    }


def verified_fact_coverage(cases: list[dict]) -> tuple[int, int]:
    """Count exact approved fact rows for the systematic ceiling questions only."""
    covered = 0
    selected = [case for case in cases if case["suite"] == "systematic"]
    chunks_by_id = {chunk["id"]: chunk for chunk in store.CHUNKS}
    for case in selected:
        slots = pipeline.analyze(case["question"])
        if not slots["fund"]:
            continue
        gold = chunks_by_id[case["evidence_id"]]
        rows = store.fact_lookup(slots["fund"], slots["tier"], None, "ceiling")
        if any(store.is_verified(row)
               and row["source"]["doc"] == gold["doc"]
               and row["source"]["page"] == gold["page"]
               and row["source"]["clause"].endswith(f"ข้อ {gold['clause']}")
               for row in rows):
            covered += 1
    return covered, len(selected)


def main() -> None:
    cases = load_cases()
    build = store.build_embeddings()
    if not store.vectors_ready():
        raise RuntimeError("BGE-M3 vectors are not ready; no fallback is used in this comparison")
    # Warm the hierarchy index before timing, as the production server does.
    store.hierarchical_search(cases[0]["question"], None)

    results: dict[str, dict[str, list[int | None]]] = {
        method: {suite: [] for suite in ("initial", "reviewed_holdout", "systematic", "all")}
        for method in METHODS
    }
    embedding_ms = []
    ranking_ms = []
    for start in range(0, len(cases), 8):
        batch = cases[start:start + 8]
        begin = time.perf_counter()
        vectors = llm.embed([case["question"] for case in batch], timeout=120)
        elapsed = (time.perf_counter() - begin) * 1000
        if len(vectors) != len(batch):
            raise ValueError("Embedding model returned the wrong number of query vectors")
        embedding_ms.extend([elapsed / len(batch)] * len(batch))
        for case, vector in zip(batch, vectors):
            begin = time.perf_counter()
            ranked = rank_methods(case["question"], vector)
            ranking_ms.append((time.perf_counter() - begin) * 1000)
            for method, rows in ranked.items():
                ids = [row["id"] for row in rows]
                rank = ids.index(case["evidence_id"]) + 1 if case["evidence_id"] in ids else None
                results[method][case["suite"]].append(rank)
                results[method]["all"].append(rank)

    print(f"model={llm.EMBED_MODEL} approved_chunks={sum(store._passes_filter(c, None) for c in store.CHUNKS)} "
          f"questions={len(cases)} vectors={build['chunks']} embedded_now={build['embedded_now']}")
    for suite in ("initial", "reviewed_holdout", "systematic", "all"):
        print(f"[{suite}]")
        for method in METHODS:
            metrics = score(results[method][suite])
            print(f"  {method}: hit@1={metrics['hit1']}/{metrics['total']} "
                  f"hit@3={metrics['hit3']}/{metrics['total']} "
                  f"hit@5={metrics['hit5']}/{metrics['total']} MRR@5={metrics['mrr5']:.3f}")
    exact, total = verified_fact_coverage(cases)
    print(f"approved structured ceiling facts available: {exact}/{total} systematic questions")
    print(f"mean query embedding={statistics.mean(embedding_ms):.1f} ms; "
          f"mean ranking for all compared methods={statistics.mean(ranking_ms):.1f} ms")
    print("These figures measure evidence retrieval, not answer correctness or web latency.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
