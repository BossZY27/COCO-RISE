# RISE MVP runbook

Prerequisites: Docker Engine with Compose, CPU/RAM for Qwen3 1.7B and BGE-M3, free disk for model/image volumes. The four original PDF copies in `funds-data/` are seed inputs, never auto-approved by setup.

```sh
python scripts/migrate_env.py
# Set RISE_ADMIN_TOKEN in .env to enable admin operations.
docker compose up --build -d
# Optional model installation; the API works with verified fallback while unavailable.
docker compose --profile setup run --rm models
docker compose exec api python scripts/model_smoke.py
```

Chat: http://127.0.0.1:8077/ (v1). Interactive Swagger: http://127.0.0.1:8077/docs. Read-only v2 contract: http://127.0.0.1:8077/design/docs. Readiness: `/v2/ready`; dependency status: `/v2/status`.

Compose runs migration then insert-only seed before API/worker startup. PostgreSQL, PDFs and Ollama models have separate named volumes. Only the API is exposed, on localhost. `docker compose down` preserves volumes; do not add `-v` unless intentionally deleting local data. Restart containers with `docker compose restart api worker postgres`; readiness checks both DB connectivity and Alembic revision. An unavailable Ollama/model is degraded and falls back, not a readiness failure.

Explicit maintenance:

```sh
docker compose run --rm migrate
docker compose run --rm seed
docker compose logs --tail=100 api worker
docker compose exec api python scripts/export_openapi.py
```

To import existing quarantine records, mount the previous `data/ingestion` tree read-only at `/srv/rise/data/ingestion` for a seed run. The importer reads document/chunk/job state and original PDFs, preserving verified flags and human-edit provenance. Existing database rows always win on repeated imports. Import manifests contain source checksums; no credentials are logged.

For local Python development, install `requirements.txt`, set `RISE_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/rise`, run `alembic upgrade head`, `python scripts/seed.py`, then run `uvicorn app.main:app --port 8077` and `python -m app.workers.main` in separate processes. Empty database URL intentionally enables the legacy fixture demo only. Runtime does not recognize old environment names; the migration tool preserves existing RISE values and never prints secrets.

Defaults: `RISE_ANSWER_MODEL=qwen3:1.7b`, `RISE_EMBED_MODEL=bge-m3`, `RISE_LLM_BASE=http://ollama:11434` in Docker, one model slot, 30-second answer timeout, 2-second query embedding timeout. Change operator model settings in `.env` and recreate containers. Run `docker compose --profile setup run --rm models` after changing names. Failed index jobs may be retried through `/v2/jobs/{id}/retry` once models are installed.

The worker claims queued or expired jobs, heartbeats its lease, and limits automatic retries. Inspect page errors, correct missing OCR text through the page API, then explicitly review every page and publish. Do not retry after human edits; create a new version if extraction must be rerun. Publication queues vector indexing. BM25 remains available before a generation is complete.

Verification:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 RISE_LLM_MODE=template python -m pytest -q
python eval.py
python eval_quality.py
python scripts/export_openapi.py
# Use a dedicated PostgreSQL 17/pgvector DB; the suite clears it.
RISE_TEST_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/rise_test \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 RISE_LLM_MODE=template python -m pytest tests/test_postgres.py -q
```

The PostgreSQL suite refuses any database not named `rise_test`. It checks insert-only import, preserved draft/review gates, concurrent claims, stale-owner fencing, version publication/retraction, inline evidence atomicity, Decimal money, v1 auth/personas/upload/SSE, v2 errors/feedback/rate limits, model outage, corpus changes during generation and exact pgvector cosine retrieval. `scripts/model_smoke.py` records connectivity/completion/latency only. None of these checks establish research answer quality.

Current limits: one API process for in-memory rate limits, 15 MiB/100 pages per upload, exact BGE-M3 dimension 1024, conservative v2 model use for amount/ceiling intents. Historical citations identify PDF version/page; the source endpoint returns that PDF for browser viewing. Thai semantic verification still needs human-confirmed adversarial evaluation.
