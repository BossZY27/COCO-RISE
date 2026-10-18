"""Offline answer-path ablation using the same verified facts and answer gate."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval_quality  # noqa: E402
from app import pipeline, store  # noqa: E402
from app.retrieval_experiment import RetrievalComparison  # noqa: E402


async def main() -> None:
    cases = []
    for suite, filename in (("quality", "eval_quality.json"), ("legacy_holdout", "eval_holdout.json")):
        items = json.loads((ROOT / "data" / filename).read_text(encoding="utf-8"))
        items = eval_quality.resolve_dynamic_expectations(items, pipeline.thai_fiscal_year())
        cases.extend(({**case, "suite": suite}) for case in items)
    indices = {}
    original_retrieve = pipeline.retrieve
    results = {}

    for method in ("flat", "page_only", "parent_context", "hierarchical"):
        async def retrieve_with_method(slots):
            evidence = await original_retrieve(slots)
            if slots["intent"] not in ("list", "catalogue_details", "open_now", "recommendation"):
                year = slots["fiscal_year"]
                if year not in indices:
                    indices[year] = RetrievalComparison(store.CHUNKS, year)
                index = indices[year]
                evidence["chunks"] = getattr(index, method)(slots["standalone_question"], top_k=5)
            return evidence

        outputs = []
        with (patch.object(pipeline, "retrieve", retrieve_with_method),
              patch.object(pipeline, "_semantic", lambda query, year:
                           (store.semantic_search(query, year), "trigram"))):
            for case in cases:
                answer = await pipeline.run(case["question"], case["persona"], case.get("history"))
                outputs.append((answer, eval_quality.check(case, answer)))
        results[method] = outputs

    for method, outputs in results.items():
        for suite in ("quality", "legacy_holdout"):
            selected = [output for case, output in zip(cases, outputs) if case["suite"] == suite]
            print(f"{method} {suite}: answer checks={sum(not problems for _, problems in selected)}/{len(selected)} "
                  f"blocked={sum(answer['answer_source'] == 'blocked' for answer, _ in selected)}")
    for method in ("page_only", "parent_context", "hierarchical"):
        changed = []
        for case, (flat, flat_problems), (candidate, candidate_problems) in zip(
                cases, results["flat"], results[method]):
            if (flat["answer"] != candidate["answer"]
                    or flat["citations"] != candidate["citations"]):
                changed.append((case["id"], case["question"], flat_problems, candidate_problems))
        print(f"{method} visible answer/citation differences vs flat: {len(changed)}")
        for case_id, question, flat_problems, candidate_problems in changed:
            print(f"- {case_id} {question}: flat={flat_problems or 'pass'} "
                  f"{method}={candidate_problems or 'pass'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
