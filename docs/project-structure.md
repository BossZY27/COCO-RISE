# Project structure

RISE is a modular monolith with ports at module boundaries and adapters under `infrastructure/`. The only Python files at the top of `app/` are the public entry point, composition root, and package initializer.

```text
app/
├── __init__.py
├── main.py                         FastAPI assembly and shared lifecycle
├── bootstrap.py                    process-scoped adapter composition
├── core/                           settings, paths, security, errors, dependencies
├── modules/
│   ├── query/                      question/context analysis, templates, LLM rephrasing and verification
│   ├── knowledge/                  facts, rules, corpus, context and citations
│   ├── retrieval/                  BM25, tokenizer, RRF and semantic orchestration
│   ├── documents/                  PDF inspection, chunking, review, publication
│   ├── legacy/                     offline historical harness; not mounted by app.main
│   ├── feedback/                   request-ID feedback API
│   └── system/                     readiness and dependency status
├── infrastructure/
│   ├── postgres/                   SQLAlchemy, repositories, 4-layer DB inspector and relational schema DDL
│   ├── chroma/                     ChromaDB persistent vector store (bge-m3 1,024-dim HNSW cosine index)
│   ├── ollama/                     bounded runtime model client and offline model helpers
│   ├── files/                      private PDF storage and extraction
│   └── json/                       append-only compatibility audit adapter
├── workers/                        leased OCR, ingestion and indexing jobs
└── static/                         chat.html, admin.html and db_viewer.html
evaluation/                         offline retrieval comparison code
migrations/                         Alembic schema
scripts/                            seed, OpenAPI export and model smoke checks
funds-data/                         byte-preserved source PDFs
```

`app.main:app` and `app.workers.main` remain the deployment entry points. Runtime modules import their final locations directly; the removed top-level modules are not retained as wrappers. PostgreSQL is required for the HTTP runtime, with ChromaDB for vector retrieval. JSON supplies seed/fixture data for offline tools only. `QueryService` is the single public orchestration; the historical `engine` is retained for offline experiment regression, and shares the extracted analyzer.
