# Evaluation protocol (no results claimed)

Compare three methods from the report on the same frozen, human-confirmed question set and same published corpus revision:

1. Text-only retrieval and model answer.
2. BM25+BGE-M3 with RRF, separately selected structured facts/rules, model candidate and verification/fallback (v2).
3. The verified deterministic template path (v1/reference).

Confirm this mapping against the report before publishing results. Existing offline comparison tools in `scripts/compare_retrieval.py` and `scripts/compare_answer_paths.py` remain regression/exploration tools, not independent research results. Do not report pre-MVP demo timings as MVP results.

Freeze document-version checksums, approvals, embedding/answer model digests, prompt version, corpus revision, machine/CPU settings and temperature. Separate development/held-out sets; never tune against the final test set. Include no-answer, ambiguous alias/tier/year, pending OCR, wrong fiscal year, ceiling versus annual allocation, unsupported country/profile recommendations, cross-fund rules and prompt injection.

Report separately:

| Measure | Evidence |
|---|---|
| Retrieval recall/rank | human-selected supporting chunk/page; BM25, vector and RRF ablations |
| Answer correctness/completeness | reviewers compare claims, money units, amount type/year and retained conditions |
| Citation correctness/coverage | document/version/page/chunk supports every claim, including superseded-source cases |
| False acceptance | deliberately wrong numbers, year, fund, guarantee, negation and dropped conditions; measure gate acceptance |
| Latency | stage timings, p50/p95, cold/warm model, outage/fallback and concurrent request conditions |

Query runs retain response, candidate verification problems, model revision and prompt version. Keep rejected candidate text out of public responses. Inspect reviewer disagreement separately; automatic number/citation gates do not prove semantic entailment in Thai. A model connection smoke test is never an answer-quality evaluation. No benchmark score is asserted until the corpus, protocol and human reference answers are approved.
