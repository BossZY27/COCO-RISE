# RISE API

`/docs` is interactive Swagger for the implemented v1 and v2 endpoints. `/design/docs` shows the generated contract with submission disabled. Both use self-hosted Swagger assets allowed by the application CSP. Refresh the checked-in contract using `python scripts/export_openapi.py`; CI/tests resolve every `$ref` and validate examples against the app schemas. `RISE_EXPOSE_DOCS=off` disables all docs endpoints.

`/v1` preserves the demo request/response, personas, template answers, PDF admin flows and verified SSE. `/v2/query` has no persona or client-selected model. Model selection belongs to the operator.

| Resource | Operations |
|---|---|
| `/v2/query` | POST question/history → request_id, status, answer, answer_source, citations, verification, timings_ms |
| `/v2/feedback` | POST request_id/helpful/note; the query ID must exist |
| `/v2/status`, `/v2/ready` | GET service state and database/migration readiness |
| `/v2/documents` | GET list; POST raw application/pdf, optional title/fiscal_year_be query parameters and X-PDF-Name header |
| `/v2/documents/{id}` | GET metadata and versions |
| `/v2/documents/{id}/versions` | GET versions; POST raw PDF as a new immutable version |
| `/v2/versions/{id}` | GET metadata |
| `/v2/versions/{id}/pages`, `/chunks` | GET extracted drafts |
| `/v2/versions/{id}/pages/{page}` | PATCH text and reviewer; invalidates text review |
| `/v2/versions/{id}/review` | POST reviewer/source_sha256/confirmed_pages; checks every page |
| `/v2/versions/{id}/publish`, `/withdraw` | POST reviewer; changes current retrieval atomically |
| `/v2/versions/{id}/source` | GET private PDF for admin review |
| `/v2/sources/{version_id}/pages/{page}` | GET published historical PDF; citations remain resolvable after supersede/withdraw |
| `/v2/jobs`, `/{job_id}` | GET jobs/status; POST `/{job_id}/retry` for failed jobs |
| `/v2/facts`, `/v2/rules` | GET list; POST draft with inline evidence references |
| `/v2/facts/{id}`, `/v2/rules/{id}` | GET item; PATCH full draft replacement |
| `/v2/facts/{id}/review`, `/v2/rules/{id}/review` | POST reviewer and decision (`approved` or `rejected`) |

Document, knowledge and job endpoints require `X-Admin-Token` (set `RISE_ADMIN_TOKEN`). Published source citations, query, feedback and service state are public. Empty tokens disable admin access. Admin responses and query results use no-store.

A knowledge draft accepts evidence `{version_id, page, chunk_id?, quote, clause?}`. IDs and source content are validated and evidence is inserted in the same transaction. Approval additionally requires current published evidence. Text review does not approve facts/rules. Approved knowledge cannot be overwritten; corrected source content requires a new version and new knowledge draft.

Query citations contain `document_id`, immutable `version_id`, `page`, `chunk_id`, quote and URL. Superseded or withdrawn versions remain available through their citation URL for audit, while retrieval selects only the active current generation.

Query HTTP 200 states: `answered`, `clarification_required`, `insufficient_evidence`, `blocked`. Server errors never include database credentials. HTTP errors: 401 admin auth, 404 missing/private resource, 409 state conflict, 413 oversized upload, 415 wrong content type, 422 validation, 429 rate limit (Retry-After), 503 required database/schema unavailable.
