"""Grounded orchestration; application policy has no database or HTTP imports."""
import time
import uuid
from app.core import security as limits
from app.modules.knowledge import corpus as store
from app.modules.query import answers, engine as pipeline
from app.modules.query.composer import compose_answer
from app.modules.knowledge.context import use_corpus
from app.modules.knowledge.rules import describe_rules
from app.modules.knowledge.formatting import cite
from app.modules.query.verification import verify


def verify_candidate(candidate, template, citations, used, evidence, slots, ruling):
    result = verify(candidate, citations, used, evidence, slots, slots["intent"] != "list", ruling)
    # shortcut: arbitrary Thai entailment is not provable by rules; require all
    # reviewed conditions verbatim until a human-confirmed evaluation supports relaxing this gate.
    conditions = [f.get("note") for f in used if f.get("note")]
    for condition in conditions:
        if condition not in candidate:
            result["problems"].append("Candidate omitted a reviewed condition")
    import re
    numbers = lambda value: set(re.findall(r"(?<!\d)\d[\d,]*(?:\.\d+)?", value))
    if numbers(template) != numbers(candidate):
        result["problems"].append("Candidate changed or omitted a number/year")
    for f in used:
        if f["fund"] not in candidate or (f.get("tier") and f["tier"] not in candidate):
            result["problems"].append("Candidate omitted fund or tier")
    for phrase in ("เพดาน", "ยังไม่มีข้อมูลวงเงินจัดสรร", "ไม่ใช่วงเงินจัดสรร", "รอ", "อนุมัติ", "คนละ", "จำนวนทุนรวม"):
        if phrase in template and phrase not in candidate:
            result["problems"].append("Candidate omitted a qualification: " + phrase)
    if any(w in candidate for w in ("แน่นอน", "รับประกัน", "ได้รับสิทธิ์", "ได้รับแน่", "ได้รับเงิน")):
        result["problems"].append("Candidate asserted an unsupported guarantee")
    result["ok"] = not result["problems"]
    return result


def citation_records(corpus, used, chunks, labels):
    records = []
    for fact in used:
        records.extend(corpus.citations.get(fact["id"], []))
    for chunk in ([] if used else chunks):
        if cite(chunk) in labels or any(chunk["doc"] in label and f"หน้า {chunk['page']}" in label for label in labels):
            records.append({"document_id": chunk["document_id"], "version_id": chunk["version_id"],
                            "page": chunk["page"], "chunk_id": chunk["chunk_id"], "quote": chunk["text"],
                            "label": chunk["doc"], "clause": chunk.get("clause", ""),
                            "url": f"/v2/sources/{chunk['version_id']}/pages/{chunk['page']}"})
    seen = set()
    out = []
    for r in records:
        key = (r["version_id"], r["page"], r["chunk_id"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


class QueryService:
    def __init__(self, corpus_repository, repository, retrieval, answer_model):
        self.corpora, self.repository, self.retrieval, self.answer_model = corpus_repository, repository, retrieval, answer_model

    def run(self, question, history):
        started = time.perf_counter()
        times = {}
        corpus = self.corpora.load()
        with use_corpus(corpus):
            slots = pipeline.analyze(question, history)
            times["analyze"] = (time.perf_counter() - started) * 1000
            t = time.perf_counter()
            chunks, method = self.retrieval.search(slots.get("query") or question, slots["fiscal_year"], corpus)
            kind = "ceiling" if slots["intent"] == "ceiling" else "allocated"
            rows = store.fact_lookup(slots["fund"], slots["tier"], slots["fiscal_year"], kind)
            pending = any(not store.is_verified(f) for f in rows)
            facts = [f for f in rows if store.is_verified(f)]
            fallback = False
            if not facts and kind == "allocated" and slots["intent"] in {"amount", "eligibility"}:
                facts = [f for f in store.fact_lookup(slots["fund"], slots["tier"], None, "ceiling") if store.is_verified(f)]
                fallback = bool(facts) and not pending
            evidence = {"chunks": chunks, "facts": facts, "pending_review": pending,
                        "ceiling_fallback": fallback, "semantic_by": method}
            times["retrieve"] = (time.perf_counter() - t) * 1000
            # v2 is public document QA; no persona-based eligibility assertions.
            ruling = {"verdict": "no_rule", "reason": None, "source": None, "missing": ["staff_type", "department"]}
            t = time.perf_counter()
            template, labels, used = compose_answer(slots, evidence, ruling)
            selected_rules = (store.rules_for(slots["fund"], slots["fiscal_year"])
                              if slots["intent"] == "eligibility" or slots.get("criteria_request") else [])
            described = describe_rules(selected_rules)
            if described:
                template, labels = described
                used = []
                evidence["chunks"] = [*chunks, {"text": template}]
            before, after = answers.name_notes(slots)
            template = " ".join(part for part in (before, template, after) if part)
            check = verify(template, labels, used, evidence, slots, slots["intent"] != "list", ruling)
            times["template"] = (time.perf_counter() - t) * 1000
            answer, source, model_revision, candidate_problems = template, "template", None, []
            if check["ok"] and used and slots["intent"] in {"amount", "ceiling"}:
                t = time.perf_counter()
                try:
                    candidate, model_revision = self.answer_model.candidate(template, labels)
                    candidate_check = verify_candidate(candidate, template, labels, used, evidence, slots, ruling)
                    if candidate_check["ok"]:
                        answer, source = candidate, "llm"
                    else:
                        candidate_problems = candidate_check["problems"]
                except Exception as exc:
                    candidate_problems = [f"Answer model unavailable: {type(exc).__name__}"]
                times["llm"] = (time.perf_counter() - t) * 1000
            citations = citation_records(corpus, used, corpus.CHUNKS, labels)
            if described:
                citations = [ref for rule in selected_rules for ref in corpus.citations.get(rule["id"], [])]
            status = "answered" if labels and citations else "insufficient_evidence"
            if slots["intent"] in {"unknown", "clarify_reference"} or slots.get("alternatives") or (not slots.get("fund") and slots["intent"] == "amount"):
                status = "clarification_required"
            if not check["ok"] or (labels and not citations):
                status, source, answer, citations = "blocked", "blocked", "ข้อมูลไม่เพียงพอ คำตอบไม่ผ่านการตรวจหลักฐาน", []
                check = {"ok": False, "problems": check["problems"] or ["Missing versioned citation"]}
            # A revision change during model generation invalidates the whole candidate.
            if self.repository.corpus_revision() != corpus.revision:
                status, source, answer, citations = "blocked", "blocked", "หลักฐานมีการเปลี่ยนแปลง กรุณาถามใหม่", []
                check = {"ok": False, "problems": ["Corpus changed while answering"]}
            times["total"] = (time.perf_counter() - started) * 1000
            identifier = uuid.uuid4()
            result = {"request_id": str(identifier), "status": status, "answer": answer, "answer_source": source,
                      "citations": citations, "verification": {**check, "candidate_problems": candidate_problems},
                      "timings_ms": {k: round(v, 2) for k, v in times.items()}}
        self.repository.save_query(identifier, question, result, model_revision)
        return result
