"""One grounded orchestration for the API: select, compose, rephrase, verify."""
import re
import time
import uuid
from collections import Counter
from app.modules.knowledge import corpus as store
from app.modules.knowledge.context import use_corpus
from app.modules.knowledge.formatting import cite
from app.modules.knowledge.rules import describe_rules
from app.modules.query import analysis, answers
from app.modules.query.composer import compose_answer
from app.modules.query.verification import verify, QUALITATIVE_REQUIREMENT

QUERY_BUDGET = 35.0
REMEMBERED_INTENTS = {"amount", "ceiling", "tiers", "details", "definition", "eligibility", "application"}
REPHRASE_INTENTS = REMEMBERED_INTENTS | {"compare", "rank", "catalogue_details"}


def verify_candidate(candidate, template, citations, used, evidence, slots, ruling):
    result = verify(candidate, citations, used, evidence, slots, slots["intent"] != "list", ruling)
    numbers = lambda value: Counter(re.findall(r"(?<!\d)\d[\d,]*(?:\.\d+)?", value))
    if numbers(template) != numbers(candidate):
        result["problems"].append("Candidate changed or omitted a number/year")
    # shortcut: rules cannot prove Thai entailment; preserve each factual clause
    # verbatim until human-confirmed adversarial evaluation supports freer rewriting.
    for clause in re.split(r"[;\n]|(?<=[.!?])\s+", template):
        clause = clause.strip()
        if clause and (re.search(r"\d|ต้อง|เงื่อนไข|ไม่ใช่|ยังไม่มี|ไม่เกิน|สูงสุด|เพดาน|ตัดสินสิทธิ์", clause) or not used):
            if clause not in candidate:
                result["problems"].append("Candidate changed a protected factual clause")
    for claim in QUALITATIVE_REQUIREMENT.findall(candidate):
        if claim not in template:
            result["problems"].append("Candidate introduced an unsupported eligibility condition")
    for fact in used:
        if fact["fund"] not in candidate or (fact.get("tier") and fact["tier"] not in candidate):
            result["problems"].append("Candidate omitted fund or tier")
        if fact.get("note") and fact["note"] not in candidate:
            result["problems"].append("Candidate omitted a reviewed condition")
    if any(w in candidate for w in ("แน่นอน", "รับประกัน", "ได้รับสิทธิ์", "ได้รับแน่", "ได้รับเงิน")):
        result["problems"].append("Candidate asserted an unsupported guarantee")
    result["ok"] = not result["problems"]
    return result


def citation_records(corpus, used, chunks, labels):
    def covered(source):
        # Only catalogue labels intentionally cite a whole annex page.
        return f"{source.get('doc', '')} บัญชีแนบท้าย หน้า {source.get('page')}" in labels

    records = []
    # Catalogue/detail renderers may return labels without a used-fact list.
    rows = used or [r for r in [*corpus.FACTS, *corpus.RULES]
                    if store.is_verified(r) and r.get("source") and (cite(r["source"]) in labels or covered(r["source"]))]
    for row in rows:
        records.extend(corpus.citations.get(row["id"], []))
    for chunk in chunks:
        if chunk.get("verified") is True and chunk.get("status") == "active" and (cite(chunk) in labels or covered(chunk)):
            records.append({"document_id": chunk["document_id"], "version_id": chunk["version_id"],
                            "page": chunk["page"], "chunk_id": chunk["chunk_id"], "quote": chunk["text"],
                            "label": chunk["doc"], "clause": chunk.get("clause", ""),
                            "url": f"/api/sources/{chunk['version_id']}/pages/{chunk['page']}"})
    out, seen = [], set()
    for ref in records:
        key = (ref["version_id"], ref["page"], ref.get("chunk_id"))
        if key not in seen:
            seen.add(key)
            out.append({**ref, "url": f"/api/sources/{ref['version_id']}/pages/{ref['page']}"})
    return out


class QueryService:
    def __init__(self, corpus_repository, repository, retrieval, answer_model):
        self.corpora, self.repository, self.retrieval, self.answer_model = corpus_repository, repository, retrieval, answer_model

    def run(self, question, history, context=None):
        started = time.monotonic()
        deadline = started + QUERY_BUDGET
        remaining = lambda: max(0.0, deadline - time.monotonic())
        times = {}
        settings = self.answer_model.settings
        processing = {"slot_attempted": False, "slot_model": None, "slot_fallback_reason": None,
                      "answer_attempted": False, "answer_model": None, "answer_fallback_reason": None}

        def extract_slots(text, known_fund=None):
            if remaining() <= 0:
                processing["slot_fallback_reason"] = "query_deadline"
                return None
            processing.update(slot_attempted=True, slot_model=settings.answer_model)
            try:
                return self.answer_model.extract_slots(text, known_fund=known_fund, timeout=min(5, remaining()))
            except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                processing["slot_fallback_reason"] = type(exc).__name__
                return None
            except Exception as exc:
                # Optional model failures must not bypass deterministic clarification.
                processing["slot_fallback_reason"] = type(exc).__name__
                return None

        corpus = self.corpora.load()
        with use_corpus(corpus):
            t = time.monotonic()
            slots = analysis.analyze(question, history, context=context, use_llm=settings.slot_llm_fallback,
                                     slot_extractor=extract_slots, default_year=settings.fiscal_year)
            times["analyze"] = (time.monotonic() - t) * 1000
            processing.update(analysis_method=slots["method"], corpus_revision=corpus.revision)
            t = time.monotonic()
            no_search = slots["intent"] in {"list", "catalogue_details", "open_now", "recommendation", "clarify_reference", "unknown", "smalltalk"}
            if no_search:
                chunks, method, retrieval_reason = [], "not_needed", None
            else:
                chunks, method, retrieval_reason = self.retrieval.search(
                    slots["standalone_question"], slots["fiscal_year"], corpus, timeout=remaining())
            processing.update(retrieval_method=method, retrieval_fallback_reason=retrieval_reason)
            kind = "ceiling" if slots["intent"] == "ceiling" else "allocated"
            rows = store.fact_lookup(slots["fund"], slots["tier"], slots["fiscal_year"], kind) if slots["fund"] else []
            pending = any(not store.is_verified(f) for f in rows)
            facts = [f for f in rows if store.is_verified(f)]
            fallback = False
            if not facts and kind == "allocated" and slots["intent"] in {"amount", "eligibility"}:
                facts = [f for f in store.fact_lookup(slots["fund"], slots["tier"], None, "ceiling") if store.is_verified(f)]
                fallback = bool(facts) and not pending
            selected_rules = store.rules_for(slots["fund"], slots["fiscal_year"]) if slots["fund"] and slots["intent"] in {"eligibility", "details"} else []
            selected_rules = [r for r in selected_rules if store.is_verified(r)]
            evidence = {"chunks": chunks, "facts": facts, "rules": selected_rules,
                        "pending_review": pending, "ceiling_fallback": fallback, "semantic_by": method}
            times["retrieve"] = (time.monotonic() - t) * 1000
            ruling = {"verdict": "not_evaluated", "reason": None, "source": None, "missing": []}
            t = time.monotonic()
            template, labels, used = compose_answer(slots, evidence, ruling)
            described = describe_rules(selected_rules)
            if described:
                rule_text, rule_labels = described
                if slots["intent"] == "eligibility":
                    template, labels, used = rule_text, rule_labels, []
                else:
                    template, labels = template + " " + rule_text, list(dict.fromkeys([*labels, *rule_labels]))
            before, after = answers.name_notes(slots)
            template = " ".join(part for part in (before, template, after) if part)
            citations = citation_records(corpus, used, corpus.CHUNKS, labels)
            for rule in selected_rules:
                for ref in corpus.citations.get(rule["id"], []):
                    ref = {**ref, "url": f"/api/sources/{ref['version_id']}/pages/{ref['page']}"}
                    if ref not in citations:
                        citations.append(ref)
            # Verifier receives only actual source chunks and reviewed structured rows.
            referenced = {(str(r["version_id"]), r["page"]) for r in citations}
            evidence["chunks"] = [c for c in corpus.CHUNKS if c.get("verified") is True and c.get("status") == "active"
                                  and (str(c["version_id"]), c["page"]) in referenced]
            check = verify(template, labels, used, evidence, slots, slots["intent"] not in {"list", "catalogue_details"}, ruling)
            times["template"] = (time.monotonic() - t) * 1000
            status = "answered" if labels and citations else "insufficient_evidence"
            if slots["intent"] in {"unknown", "clarify_reference"} or slots.get("alternatives") or (not slots["fund"] and slots["intent"] in {"amount", "ceiling"}):
                status = "clarification_required"
            answer, source, model_revision, candidate_problems = template, "template", None, []
            eligible = status == "answered" and check["ok"] and slots["intent"] in REPHRASE_INTENTS
            if not settings.answer_enabled:
                processing["answer_fallback_reason"] = "disabled"
            elif not eligible:
                processing["answer_fallback_reason"] = "not_applicable"
            elif remaining() <= 0:
                processing["answer_fallback_reason"] = "query_deadline"
            else:
                processing.update(answer_attempted=True, answer_model=settings.answer_model)
                t = time.monotonic()
                try:
                    candidate, model_revision = self.answer_model.candidate(template, labels, timeout=remaining())
                    if remaining() <= 0:
                        raise TimeoutError("Query deadline reached")
                    candidate_check = verify_candidate(candidate, template, labels, used, evidence, slots, ruling)
                    if candidate_check["ok"]:
                        answer, source = candidate, "llm"
                    else:
                        candidate_problems = candidate_check["problems"]
                        processing["answer_fallback_reason"] = "verification_failed"
                except Exception as exc:
                    candidate_problems = [f"Answer model unavailable: {type(exc).__name__}"]
                    processing["answer_fallback_reason"] = type(exc).__name__
                times["llm"] = (time.monotonic() - t) * 1000
            if not check["ok"] or (labels and not citations):
                status, source, answer, citations = "blocked", "blocked", "ข้อมูลไม่เพียงพอ คำตอบไม่ผ่านการตรวจหลักฐาน", []
                check = {"ok": False, "problems": check["problems"] or ["Missing versioned citation"]}
            if self.repository.corpus_revision() != corpus.revision:
                status, source, answer, citations = "blocked", "blocked", "หลักฐานมีการเปลี่ยนแปลง กรุณาถามใหม่", []
                check = {"ok": False, "problems": ["Corpus changed while answering"]}
            remembered = None
            if check["ok"] and source != "blocked" and slots["fund"] and slots["intent"] in REMEMBERED_INTENTS and not slots.get("alternatives"):
                remembered = {k: slots[k] for k in ("fund", "tier", "fiscal_year", "year_assumed", "intent")}
            times["total"] = (time.monotonic() - started) * 1000
            identifier = uuid.uuid4()
            result = {"request_id": str(identifier), "status": status, "answer": answer, "answer_source": source,
                      "citations": citations, "verification": {**check, "candidate_problems": candidate_problems},
                      "timings_ms": {k: round(v, 2) for k, v in times.items()}, "context": remembered,
                      "processing": processing}
        self.repository.save_query(identifier, question, result, model_revision)
        return result
