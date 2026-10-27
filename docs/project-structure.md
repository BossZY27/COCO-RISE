# Project structure

RISE is a modular monolith with ports at module boundaries and adapters under `infrastructure/`. The only Python files at the top of `app/` are the public entry point, composition root, and package initializer.

```text
app/
├── __init__.py
├── main.py                         FastAPI assembly and shared lifecycle
├── bootstrap.py                    process-scoped adapter composition
├── core/                           settings, paths, security, errors, dependencies
├── modules/
│   ├── query/                      analysis, persona lookup, templates, typed claims, LLM RAG and verification
│   ├── knowledge/                  facts, rules, corpus, context and citations
│   ├── retrieval/                  BM25, tokenizer, RRF and semantic orchestration
│   ├── documents/                  PDF inspection, chunking, review, publication
│   ├── legacy/                     stable v1 routes, schemas and admin API
│   ├── feedback/                   v2 feedback API
│   └── system/                     readiness and dependency status
├── infrastructure/
│   ├── postgres/                   SQLAlchemy, repositories, 4-layer DB inspector and relational schema DDL
│   ├── chroma/                     ChromaDB persistent vector store (bge-m3 1,024-dim HNSW cosine index)
│   ├── ollama/                     bounded model clients and v1 compatibility
│   ├── files/                      private PDF storage and extraction
│   └── json/                       append-only compatibility audit adapter
├── workers/                        leased OCR, ingestion and indexing jobs
└── static/                         chat.html, admin.html and db_viewer.html
evaluation/                         offline retrieval comparison code
migrations/                         Alembic schema
scripts/                            seed, OpenAPI export and model smoke checks
funds-data/                         byte-preserved source PDFs
```

`app.main:app` and `app.workers.main` remain the deployment entry points. Runtime modules import their final locations directly; the removed top-level modules are not retained as wrappers. PostgreSQL is the primary runtime when `RISE_DATABASE_URL` is configured. The JSON adapter supplies compatibility behavior and fixtures without becoming a dependency of the v2 HTTP layer.
