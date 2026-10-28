> Updated 28 October 2026: the approved implementation uses one `/api/*` contract and QueryService, PostgreSQL + ChromaDB, and verified LLM rephrasing/template fallback. Legacy HTTP contracts below are historical decisions, not mounted routes.

# ADR 001: Modular monolith with ports and adapters

Accepted for the RISE API MVP, 20 October 2026.

The existing Thai question analyzer, renderer, verification and v1 contract are valuable. Keep them, introduce request-bound corpus snapshots, and move persistence/model/file operations behind adapters. Separate API and worker processes share the same image and migrations. This provides explicit boundaries without service-to-service deployment overhead.

Modules: query (analysis, evidence selection, candidate verification), knowledge (facts/rules/aliases/personas), documents (version/review/publication/jobs), retrieval (BM25, vector, RRF), feedback (query-linked reports), and system (readiness/degraded service status). Infrastructure owns PostgreSQL transactions, Ollama HTTP and private PDF files. Legacy facade modules remain compatibility surfaces; new HTTP routes contain validation and delegation.

The schema uses UUIDs, preserved `legacy_id`, Decimal/numeric money, Gregorian timestamps, and explicit `fiscal_year_be` for Buddhist fiscal years. Original JSON/PDFs remain seed evidence. Import is insert-only: it preserves existing approval booleans and document-gated legacy rules without assigning a new human reviewer.

Related research supplied with the implementation plan:

- [Su and Li, 2024](https://arxiv.org/html/2401.11867v1): a systematic grey literature review informing modular-monolith boundaries, not proof of universal superiority.
- [Nugroho et al., 2022](https://research.monash.edu/en/publications/clean-architecture-implementation-impacts-on-maintainability-aspe/): a case study motivating separation of backend policy from infrastructure; RISE maintainability must be evaluated independently.
- [Parnas, 1972](https://doi.org/10.1145/361598.361623): information hiding informs boundaries around storage, OCR and models.

The plan explicitly authorizes a free-form candidate path in v2, superseding the earlier handoff restriction for that path. v1 stays deterministic. The v2 gate checks numbers, years, fund/tier names, amount qualifications, reviewed notes and citations, with full-template fallback on timeout, truncation or failure. These rules cannot establish arbitrary Thai semantic entailment. Human-confirmed adversarial evaluation is required before interpreting model acceptance as answer quality.
