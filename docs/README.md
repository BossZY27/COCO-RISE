# RISE implementation docs

- [Runbook](runbook.md): Docker startup, migrations, import, model setup, tests and recovery.
- [Architecture](architecture.md) and [decision record](adr-001-modular-monolith.md).
- [API guide](api.md) and [generated API contract](openapi-mvp.json).
- [Pipeline](pipeline.md), [data model](data-model.md), [project structure](project-structure.md).
- [Evaluation protocol](evaluation.md): separate retrieval, answer, citation and latency measures.

The Docker deployment uses PostgreSQL 17 and ChromaDB. All HTTP endpoints use `/api/*`; missing database configuration returns 503 instead of a JSON demo. Legacy modules and JSON fixtures are retained only for offline regression/evaluation. The proposal is a research plan, not evidence of answer quality or completed human review.
