> สถานะ runtime ปัจจุบัน (28 ตุลาคม 2569): ผู้ใช้อนุมัติการรวมเป็น `/api/*` และให้ LLM เรียบเรียงจากหลักฐานพร้อม verifier/template fallback ใช้ PostgreSQL + ChromaDB ตาม [architecture](docs/architecture.md) และ [runbook](docs/runbook.md) เนื้อหาด้านล่างเป็นบันทึกเดิม; กฎ human review, source grounding และการแยกประเภทวงเงินยังใช้เหมือนเดิม

> Historical demo/research context. The implementation plan now authorizes the PostgreSQL MVP and the verified v2 candidate path; v1 remains deterministic. See [current architecture](docs/architecture.md) and [runbook](docs/runbook.md). This update does not approve any source data.

# RISE Proposal Brief From Annotated PDF

Source: `C:\Users\Lenovo\Downloads\RISE-เค้าโครงโครงงานคอมพิวเตอร์-2569_260929_032046.pdf` (9 pages, annotated in red). The red handwriting is part of the page image; text extraction alone misses it. This file records the reading used for proposal edits, not new grant facts or permission to approve OCR data.

## Scope and Claim Boundaries

- The proposal's research focus is grounded Thai-language question answering over four KKU research-fund documents. The method combines keyword retrieval, semantic-vector retrieval, and structured facts/rules, with citations and a verification step.
- The reviewer crossed out profile-based grant matching in objective 1 and draft-proposal completeness checking in objective 3. Do not describe either as a committed deliverable in the revised proposal. Personal eligibility remains subject to the verified rules and must not be guaranteed by the bot.
- Offline preparation includes OCR for scanned PDFs, human comparison with original pages, document metadata, chunks, facts, and rules. Unreviewed OCR or annual allocation figures stay unavailable to answers, as required by `HANDOFF.md`.
- The comparison is text-only RAG versus the hybrid method on the same reviewer-confirmed questions. Report retrieval, answer, citation, and latency measures separately. A passing code test is not a research result.
- The current demo uses JSON and in-memory vector data. PostgreSQL with pgvector is a proposed target architecture, not a completed implementation. The university personnel database is not connected.
- Remove unmeasured timings and unsupported user-benefit claims from the architecture figure. The proposal must not say the system increases a person's chance of winning a grant.

## Page-by-Page Review Notes

| PDF page | Reviewer mark | Editorial response |
|---|---|---|
| 3 | Mark near “RISE” in the rationale | Explain the project name and make the title consistent throughout; do not invent an expansion for the acronym. |
| 4 | Objective 1 struck through, note that matching may not be feasible | Remove grant matching from the committed objectives. |
| 4 | Draft-proposal checking struck through in objective 3 | Remove it from methods, scope, schedule, and benefits as well. |
| 4 | Notes about missing vector/structured-KB literature and OCR | Add directly relevant primary references and explain their actual role in this design. |
| 5 | Architecture marked for replacement | Redraw as offline preparation plus online evidence-based answer path; no unsupported timing numbers. |
| 6 | Mark OCR explicitly; strike advisor co-definition wording; mark the web/API description for revision | Describe OCR, human review, FastAPI, and the real demo-versus-target distinction precisely. |
| 7 | Strike “จับคู่ทุน” in the user and system scope | Keep the scope to evidence-backed retrieval and question answering. |
| 8 | Handwritten schedule note appears to say work spans 2569–2571; references note asks for vector and structured-KB sources | The revised proposal uses a provisional 2569–2571 phase grid, explicitly marked as needing advisor confirmation; add verified sources. |
| 9 | Advisor form title struck through and labeled wrong | Make it identical to the final Thai title on cover and project-title page. |

The original PDF includes two cover pages, one of which renders blank. Do not reproduce that artifact in the revised DOCX. The revised title follows the annotated PDF's typed title, but the red "แก้ชื่อ" mark is not legible enough to establish a replacement title. Ask the owner/advisor to confirm the final wording. The 2569–2571 schedule is provisional because the handwriting does not supply month-by-month milestones.

## Current Demo Versus Proposal

For factual status, consult `HANDOFF.md`, `README.md`, and current tests. Do not copy numerical status counts into the proposal without rerunning or checking the primary data. The proposal is an academic plan; the demo is a partial implementation. Updating the proposal does not itself authorize changing production data, setting `verified` to true, connecting university records, or promising matching and proposal checking.
