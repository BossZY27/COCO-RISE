"""Compare flat and page-hierarchical retrieval on fixed approved evidence."""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.modules.knowledge import corpus as store  # noqa: E402
from evaluation.retrieval import RetrievalComparison  # noqa: E402


def systematic_cases(index: RetrievalComparison) -> list[dict]:
    """One fixed query pattern per approved central-criteria fact row."""
    by_source = {}
    for chunk in index.chunks:
        by_source.setdefault((chunk["doc"], chunk["page"], chunk.get("clause")), chunk["id"])
    cases = []
    for fact in store.FACTS:
        if fact["amount_type"] != "ceiling" or not store.is_verified(fact):
            continue
        source = fact["source"]
        clause = source["clause"].removeprefix("บัญชีแนบท้าย ข้อ ")
        evidence_id = by_source.get((source["doc"], source["page"], clause))
        if evidence_id is None:
            raise ValueError(f"No approved chunk for fact {fact['id']}")
        tier = f" {fact['tier']}" if fact.get("tier") else ""
        cases.append({"question": f"เพดานตามหลักเกณฑ์กลางของ{fact['fund']}{tier}เท่าไร",
                      "evidence_id": evidence_id, "category": "ceiling_fact"})
    return cases


def main() -> None:
    index = RetrievalComparison(store.CHUNKS)
    suites = {
        "initial": json.loads((ROOT / "data" / "retrieval_benchmark.json").read_text(encoding="utf-8")),
        "holdout": json.loads((ROOT / "data" / "retrieval_benchmark_holdout.json").read_text(encoding="utf-8")),
        "systematic": systematic_cases(index),
    }
    approved_ids = {chunk["id"] for chunk in index.chunks}
    cases = [(suite, case) for suite, items in suites.items() for case in items]
    missing = [case["evidence_id"] for _, case in cases if case["evidence_id"] not in approved_ids]
    if missing:
        raise ValueError(f"Gold evidence is not approved/active: {missing}")
    questions = [case["question"] for _, case in cases]
    if len(questions) != len(set(questions)):
        raise ValueError("Benchmark questions must be unique across both suites")

    results = {method: [] for method in ("flat", "page_only", "parent_context", "hierarchical")}
    for suite, case in cases:
        for method in results:
            hits = getattr(index, method)(case["question"], top_k=5)
            timings = []
            for _ in range(25):
                start = time.perf_counter()
                getattr(index, method)(case["question"], top_k=5)
                timings.append((time.perf_counter() - start) * 1000)
            elapsed = statistics.median(timings)
            ids = [hit["id"] for hit in hits]
            rank = ids.index(case["evidence_id"]) + 1 if case["evidence_id"] in ids else None
            results[method].append({"rank": rank, "ms": elapsed, "ids": ids,
                                    "suite": suite, "category": case.get("category", "initial")})

    print(f"Approved chunks: {len(index.chunks)} | document pages: {len(index.page_children)} "
          f"| questions: {len(cases)}")
    for suite in ("initial", "holdout", "systematic", "all"):
        print(f"[{suite}]")
        for method, rows in results.items():
            selected = [row for row in rows if suite == "all" or row["suite"] == suite]
            print(f"  {method}: hit@1={sum(row['rank'] == 1 for row in selected)}/{len(selected)} "
                  f"hit@3={sum(row['rank'] is not None and row['rank'] <= 3 for row in selected)}/{len(selected)} "
                  f"hit@5={sum(row['rank'] is not None for row in selected)}/{len(selected)} "
                  f"MRR@5={statistics.mean(1 / row['rank'] if row['rank'] else 0 for row in selected):.3f} "
                  f"median={statistics.median(row['ms'] for row in selected):.1f} ms "
                  f"p95={sorted(row['ms'] for row in selected)[int(.95 * (len(selected) - 1))]:.1f} ms")
    print("[holdout categories: hit@5]")
    for category in sorted({row["category"] for row in results["flat"] if row["suite"] == "holdout"}):
        counts = []
        for method, rows in results.items():
            selected = [row for row in rows if row["suite"] == "holdout" and row["category"] == category]
            counts.append(f"{method}={sum(row['rank'] is not None for row in selected)}/{len(selected)}")
        print(f"  {category}: {' '.join(counts)}")
    print("[misses or rank differences]")
    for position, (_, case) in enumerate(cases):
        ranks = {method: rows[position]["rank"] for method, rows in results.items()}
        if len(set(ranks.values())) > 1 or ranks["flat"] is None:
            print(f"- {case['question']}: " + " ".join(f"{name}={rank}" for name, rank in ranks.items()))


if __name__ == "__main__":
    main()
