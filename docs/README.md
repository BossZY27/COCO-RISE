# RISE implementation docs

- [Runbook](runbook.md): Docker startup, migrations, import, model setup, tests and recovery.
- [Architecture](architecture.md) and [decision record](adr-001-modular-monolith.md).
- [API guide](api.md) and [generated v2 contract](openapi-mvp.json).
- [Pipeline](pipeline.md), [data model](data-model.md), [project structure](project-structure.md).
- [Evaluation protocol](evaluation.md): separate retrieval, answer, citation and latency measures.

The Docker deployment uses PostgreSQL 17/pgvector at runtime. The empty-database-URL standalone mode retains JSON fixtures for v1 regression/evaluation. Existing proposal documents are historical research plans, not evidence of model quality or completed human data review.
