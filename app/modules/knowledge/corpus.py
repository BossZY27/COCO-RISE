"""Corpus snapshot and retrieval policy with lazy JSON fixtures for compatibility.

In the real system these become PostgreSQL tables: chunks with a BM25 index and a pgvector column,
facts and rules as ordinary rows, users and profiles as ordinary rows.
"""
import hashlib
import json
import math
import os
import re
import tempfile
import threading
from collections import Counter
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.infrastructure.ollama import legacy as llm
from app.modules.retrieval.legacy import ApprovedEvidenceIndex
from app.modules.retrieval.tokenizer import tokenize, trigrams

DATA = Path(__file__).resolve().parents[3] / "data"
EMBEDDINGS = DATA / "embeddings.json"
EMBED_BATCH = int(os.getenv("RISE_EMBED_BATCH", "16"))
EMBED_TIMEOUT = float(os.getenv("RISE_EMBED_TIMEOUT", "120"))


def _load(name: str, default=None):
    path = DATA / name
    if not path.exists() and default is not None:
        return default
    return json.loads(path.read_text(encoding="utf-8"))



def is_verified(row: dict) -> bool:
    """Trust only an explicit JSON boolean true.

    Missing markers and lookalikes such as the string ``"true"`` are pending
    review.  This keeps schema migrations and partially written imports
    fail-closed instead of silently publishing them.
    """
    return row.get("verified") is True

TIERS = ["ชั้นต้น", "ชั้นกลาง", "ชั้นสูง", "ระดับต้น", "ระดับกลาง", "ระดับสูง"]
TIER_ALIAS = {"ระดับต้น": "ชั้นต้น", "ระดับกลาง": "ชั้นกลาง", "ระดับสูง": "ชั้นสูง"}
TERMS = ["ทุน", "วิจัย", "วงเงิน", "บาท", "ปีงบประมาณ", "ปีงบ", "ปีนี้", "เพดาน", "หลักเกณฑ์กลาง", "ประกาศ",
         "ขอได้", "เท่าไร", "กี่ทุน", "สิทธิ์", "คุณสมบัติ", "ต่อทุน", "ต่อปี", "ไม่เกิน", "รวม", "คนละ",
         "ได้ไหม", "จัดสรร", "สถาบัน", "ชั้นเรียน", "ศูนย์วิจัย", "นวัตกรรม", "บูรณาการ", "งานประจำ",
         "ผู้เชี่ยวชาญ", "ต่างประเทศ", "หลังปริญญาเอก", "บุคลากร", "สังกัด"]


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _passes_filter(chunk: dict, fiscal_year: int | None) -> bool:
    """Metadata filter: only documents in force, and only the asked year or year-independent ones."""
    if chunk.get("status") != "active" or not is_verified(chunk):
        return False
    return chunk.get("fiscal_year") in (None, fiscal_year) if fiscal_year else True


def keyword_search(query: str, fiscal_year: int | None, top_k: int = 5) -> list[dict]:
    chunks, bm25, _grams = view()._SEARCH_SNAPSHOT
    if bm25 is None:
        return []
    scores = bm25.get_scores(tokenize(query, view().VOCAB))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    hits = [chunks[i] for i in ranked if scores[i] > 0 and _passes_filter(chunks[i], fiscal_year)]
    return hits[:top_k]


def semantic_search(query: str, fiscal_year: int | None, top_k: int = 5) -> list[dict]:
    """Fallback for when the embedding model is not reachable: character trigram cosine."""
    chunks, _bm25, grams = view()._SEARCH_SNAPSHOT
    q = Counter(trigrams(query))
    scored = [(_cosine(q, g), i) for i, g in enumerate(grams)]
    scored.sort(reverse=True)
    hits = [chunks[i] for s, i in scored if s > 0.02 and _passes_filter(chunks[i], fiscal_year)]
    return hits[:top_k]


def reload_published_chunks(path: Path | None = None) -> int:
    """Atomically expose only human-approved PDF passages after publication."""
    global PUBLISHED_CHUNKS, CHUNKS, _CHUNK_TOKENS, _BM25, _CHUNK_GRAMS, _SEARCH_SNAPSHOT
    source = path or DATA / "chunks_published.json"
    rows = json.loads(source.read_text(encoding="utf-8")) if source.exists() else []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("published chunks must be a list of records")
    approved = [row for row in rows if row.get("status") == "active"
                and row.get("verified") is True and row.get("published") is True]
    chunks = view().HAND_CHUNKS + view().INGESTED_CHUNKS + approved
    tokens = [tokenize(row["text"], view().VOCAB) for row in chunks]
    bm25 = BM25Okapi(tokens)
    grams = [Counter(trigrams(row["text"])) for row in chunks]
    with _DERIVED_INDEX_LOCK:
        PUBLISHED_CHUNKS = approved
        CHUNKS = chunks
        _CHUNK_TOKENS = tokens
        _BM25 = bm25
        _CHUNK_GRAMS = grams
        _SEARCH_SNAPSHOT = (chunks, bm25, grams)
        _HIERARCHY_CACHE.clear()
    return len(approved)


def hierarchical_search(query: str, fiscal_year: int | None, top_k: int = 5) -> list[dict]:
    """Search approved parent/page structure without narrowing to pages alone.

    A human approval invalidates the cached snapshot, so the next request
    sees new evidence without rebuilding the index on every question.  The
    global parent-context path backs up page selection as the corpus grows.
    """
    key = (id(view().CHUNKS), getattr(view(), "revision", None), fiscal_year)
    with _DERIVED_INDEX_LOCK:
        index = _HIERARCHY_CACHE.get(key)
        if index is None:
            index = ApprovedEvidenceIndex(view().CHUNKS, view().VOCAB, fiscal_year)
            if len(_HIERARCHY_CACHE) >= 8:
                _HIERARCHY_CACHE.clear()
            _HIERARCHY_CACHE[key] = index
    return rrf([index.hierarchical(query, top_k=top_k * 2),
                index.parent_context(query, top_k=top_k * 2)], top_k=top_k)


# ---------------------------------------------------------------- vector index (bge-m3 through Ollama)
_VECTORS: dict[str, list[float]] = {}
_EMBEDDINGS_LOCK = threading.Lock()
_DERIVED_INDEX_LOCK = threading.RLock()
_HIERARCHY_CACHE: dict[tuple[int, int | None], ApprovedEvidenceIndex] = {}


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _empty_embedding_cache() -> dict:
    return {"model": llm.EMBED_MODEL, "items": {}}


def _valid_vector(vector) -> bool:
    return (isinstance(vector, list) and bool(vector)
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in vector))


def _vector_dimension(vectors) -> int | None:
    """Return one shared positive dimension, or None for empty/mixed vectors."""
    dimensions = {len(vector) for vector in vectors if _valid_vector(vector)}
    return next(iter(dimensions)) if len(dimensions) == 1 else None


def _read_embedding_cache(force: bool) -> dict:
    """Return only cache entries that are safe to use.

    Embeddings are optional derived data. A truncated file, an older schema, or a
    cache written for another model therefore means "rebuild", not "server fails".
    """
    empty = _empty_embedding_cache()
    if force or not EMBEDDINGS.exists():
        return empty
    try:
        stored = json.loads(EMBEDDINGS.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return empty
    if not isinstance(stored, dict) or stored.get("model") != llm.EMBED_MODEL:
        return empty
    items = stored.get("items")
    if not isinstance(items, dict):
        return empty
    valid_items = {}
    for chunk_id, item in items.items():
        if (isinstance(chunk_id, str) and isinstance(item, dict)
                and isinstance(item.get("hash"), str) and _valid_vector(item.get("vec"))):
            valid_items[chunk_id] = {"hash": item["hash"], "vec": list(item["vec"])}
    if valid_items and _vector_dimension(item["vec"] for item in valid_items.values()) is None:
        return empty
    return {"model": llm.EMBED_MODEL, "items": valid_items}


def _write_embedding_cache(cache: dict) -> None:
    """Durably write a complete JSON document, then atomically swap it into place."""
    EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=EMBEDDINGS.parent,
                prefix=f".{EMBEDDINGS.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(cache, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, EMBEDDINGS)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def build_embeddings(force: bool = False, progress=None) -> dict:
    """Embed every chunk once and keep the vectors on disk. Only new or edited text is sent to the model.

    In production this becomes a pgvector column filled by the ingestion job.
    """
    global _VECTORS
    # The read/refresh/write sequence must be one critical section. Without it two
    # warm-up requests can both read an old cache and overwrite one another's work.
    with _EMBEDDINGS_LOCK:
        _VECTORS = {}
        cache = _read_embedding_cache(force)
        live_ids = {chunk["id"] for chunk in view().CHUNKS}
        cached_ids = set(cache["items"])
        cache["items"] = {chunk_id: item for chunk_id, item in cache["items"].items()
                          if chunk_id in live_ids}
        if set(cache["items"]) != cached_ids:
            _write_embedding_cache(cache)
        missing = [
            chunk for chunk in view().CHUNKS
            if cache["items"].get(chunk["id"], {}).get("hash") != _text_hash(chunk["text"])
        ]
        expected_dimension = _vector_dimension(
            item["vec"] for item in cache["items"].values()
        )
        # In batches, and saved after each one. A single request for the whole corpus times out once the
        # corpus grows, and a timeout halfway through used to throw away every vector already computed.
        for start in range(0, len(missing), EMBED_BATCH):
            batch = missing[start:start + EMBED_BATCH]
            vectors = llm.embed([chunk["text"] for chunk in batch], timeout=EMBED_TIMEOUT)
            if not isinstance(vectors, list) or len(vectors) != len(batch):
                actual = len(vectors) if isinstance(vectors, list) else 0
                raise ValueError(
                    f"Embedding model returned {actual} vectors for {len(batch)} input chunks"
                )
            if any(not _valid_vector(vector) for vector in vectors):
                raise ValueError("Embedding model returned an invalid vector")
            batch_dimension = _vector_dimension(vectors)
            if batch_dimension is None:
                raise ValueError("Embedding model returned inconsistent vector dimensions")
            if expected_dimension is not None and batch_dimension != expected_dimension:
                raise ValueError(
                    "Embedding model vector dimension does not match the existing cache; "
                    "rebuild with force=True"
                )
            expected_dimension = batch_dimension
            for chunk, vector in zip(batch, vectors):
                cache["items"][chunk["id"]] = {
                    "hash": _text_hash(chunk["text"]),
                    "vec": list(vector),
                }
            _write_embedding_cache(cache)
            if progress:
                progress(min(start + len(batch), len(missing)), len(missing))
        _VECTORS = {chunk["id"]: cache["items"][chunk["id"]]["vec"] for chunk in view().CHUNKS}
        if EMBEDDINGS == DATA / "embeddings.json":
            try:
                from app.infrastructure.chroma import vectors as chroma_vectors
                chroma_vectors.sync_corpus_chunks(list(view().CHUNKS), _VECTORS, str(cache["model"]))
            except Exception:
                pass
        return {"model": cache["model"], "chunks": len(_VECTORS), "embedded_now": len(missing)}


def vectors_ready() -> bool:
    vectors = [_VECTORS.get(chunk["id"]) for chunk in view().CHUNKS]
    return (bool(vectors) and all(_valid_vector(vector) for vector in vectors)
            and _vector_dimension(vectors) is not None)


def vector_search(query_vector: list[float], fiscal_year: int | None, top_k: int = 5) -> list[dict]:
    if not _valid_vector(query_vector):
        raise ValueError("Query embedding is not a valid vector")
    stored = [_VECTORS.get(chunk["id"]) for chunk in view().CHUNKS if _VECTORS.get(chunk["id"])]
    if not stored:
        return []
    dimension = _vector_dimension(stored)
    if dimension is None or len(query_vector) != dimension:
        raise ValueError("Query embedding dimension does not match the vector index dimension")
    if EMBEDDINGS == DATA / "embeddings.json" and dimension == 1024:
        try:
            from app.infrastructure.chroma import vectors as chroma_vectors
            chunks_by_id = {c["id"]: c for c in view().CHUNKS if _VECTORS.get(c["id"])}
            chroma_scored = chroma_vectors.search_corpus_chunks(
                query_vector, chunks_by_id, top_k=max(top_k * 2, 12)
            )
            if chroma_scored is not None:
                return [
                    chunk for score, chunk in chroma_scored
                    if score > 0.35 and _passes_filter(chunk, fiscal_year)
                ][:top_k]
        except Exception:
            pass
    norm = math.sqrt(sum(v * v for v in query_vector)) or 1.0
    scored = []
    for chunk in view().CHUNKS:
        vector = _VECTORS.get(chunk["id"])
        if not vector:
            continue
        dot = sum(a * b for a, b in zip(query_vector, vector))
        length = math.sqrt(sum(v * v for v in vector)) or 1.0
        scored.append((dot / (norm * length), chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for score, chunk in scored if score > 0.35 and _passes_filter(chunk, fiscal_year)][:top_k]


def fact_lookup(fund: str | None, tier: str | None, fiscal_year: int | None, amount_type: str) -> list[dict]:
    """Structured lookup, the part plain RAG does not have: exact rows filtered by year and amount type."""
    out = []
    for f in view().FACTS:
        if fund and f["fund"] != fund:
            continue
        if tier and f["tier"] and f["tier"] != tier:
            continue
        if f["amount_type"] != amount_type:
            continue
        if amount_type == "allocated" and fiscal_year and f["fiscal_year"] != fiscal_year:
            continue
        out.append(f)
    # A fund can have both a row for the whole fund and a row per tier. When the question named a tier,
    # the tier row answers it; the fund-wide row is the fallback that carries the yearly total.
    out.sort(key=lambda f: 0 if tier and f["tier"] == tier else 1)
    return out


# The criteria open with a definitions clause. Those sentences answer "who may apply" and
# "what does this word mean", which no amount row can answer.
# a definition ends at the next quoted term, or where the numbered clauses start again
_DEF_RE = re.compile(r"[“\"]([^”\"]{2,40})[”\"]\s*หมายความว่า\s*(.+?)(?=[“\"]|\sข้อ\s*\d|$)")


def _build_definitions() -> dict[str, dict]:
    """Read the definitions off the whole page, not off one chunk.

    The definitions clause is a long run of short entries, so chunking cuts one of them in half.
    Joining the page back together first keeps every definition whole.
    """
    pages: dict[tuple, list[dict]] = {}
    for chunk in view().CHUNKS:
        if not _passes_filter(chunk, None):
            continue
        doc, page, text = chunk.get("doc"), chunk.get("page"), chunk.get("text")
        if not isinstance(doc, str) or not isinstance(page, int) or not isinstance(text, str):
            continue
        pages.setdefault((doc, page), []).append(chunk)
    found: dict[str, dict] = {}
    for parts in pages.values():
        joined = " ".join(c["text"] for c in parts)
        for term, body in _DEF_RE.findall(joined):
            term = term.strip()
            if term not in found:
                found[term] = {"term": term, "text": " ".join(body.split()).rstrip(" ,"), "chunk": parts[0]}
    return found





def define(term: str) -> dict | None:
    return view().DEFINITIONS.get(term)


def fund_catalogue() -> list[dict]:
    """Every fund the system holds a ceiling row for, in the order the criteria list them.

    "What funds are there" is the first thing anyone asks, and it is answerable from the rows we
    already trust. Grouping follows the two headings in the annex: research grants, then activity grants.
    """
    seen: dict[str, dict] = {}
    for fact in view().FACTS:
        if (fact["amount_type"] != "ceiling" or not is_verified(fact)
                or fact["fund"] in seen):
            continue
        clause = fact["source"]["clause"].replace("บัญชีแนบท้าย ข้อ ", "")
        group = "ทุนส่งเสริมการวิจัย" if clause.startswith("1.") else "ทุนอุดหนุนกิจกรรม"
        order = [int(part) for part in clause.split(".") if part.isdigit()]
        seen[fact["fund"]] = {"fund": fact["fund"], "group": group, "clause": clause,
                              "order": order, "total_cap": fact["total_cap"],
                              "source": fact["source"]}
    return sorted(seen.values(), key=lambda f: f["order"])


_MARKS = re.compile(r"[ัิ-ฺ็-๎\s]")
FUND_CLAUSE = re.compile(r"^\d+\.\d")
ANNEX_DOC = "หลักเกณฑ์กลาง"


def _annex_index() -> tuple[dict[str, list[str]], set]:
    """Which funds each clause of the criteria annex sets, and the annex pages, from the reviewed rows.
    Every row records the clause it was read from, so this link is data a person checked."""
    funds: dict[str, list[str]] = {}
    pages = set()
    for fact in view().FACTS:
        source = fact.get("source") or {}
        if not is_verified(fact) or source.get("doc") != ANNEX_DOC:
            continue
        clause = source.get("clause", "").replace("บัญชีแนบท้าย ข้อ ", "")
        funds.setdefault(clause, [])
        if fact["fund"] not in funds[clause]:
            funds[clause].append(fact["fund"])
        pages.add(source.get("page"))
    return funds, pages





def refresh_derived_indexes() -> dict[str, int]:
    """Rebuild views derived from live facts/chunks after human approval.

    Readers continue using the previous complete snapshots while the small
    indexes are rebuilt.  The annex pair is published as one tuple so a query
    never combines clause mappings from one refresh with page metadata from
    another.
    """
    global DEFINITIONS, ANNEX_FUNDS, ANNEX_PAGES, _ANNEX_INDEX
    with _DERIVED_INDEX_LOCK:
        _HIERARCHY_CACHE.clear()
        definitions = _build_definitions()
        annex_funds, annex_pages = _annex_index()
        DEFINITIONS = definitions
        ANNEX_FUNDS = annex_funds
        ANNEX_PAGES = annex_pages
        _ANNEX_INDEX = (annex_funds, annex_pages)
        return {"definitions": len(definitions), "annex_clauses": len(annex_funds)}


def funds_in_clause(chunk: dict) -> list[str] | None:
    """The funds an annex clause sets, read from the clause numbers the reviewed rows cite. A sub-clause
    such as 2.1.1 belongs to the fund of 2.1. None when the chunk is no annex clause; [] when no reviewed
    row covers it, as for the group 3 funds of 3.1 and 3.2, which the system does not hold."""
    clause = chunk.get("clause", "")
    annex_funds, annex_pages = view()._ANNEX_INDEX
    if chunk.get("doc") != ANNEX_DOC or chunk.get("page") not in annex_pages or not FUND_CLAUSE.match(clause):
        return None
    while clause:
        if clause in annex_funds:
            return list(annex_funds[clause])
        clause = clause.rpartition(".")[0]
    return []


def _skeleton(text: str) -> str:
    """Drop the marks above and below the line plus spaces. The annex writes a fund one way and the
    announcement another, and OCR adds its own drift; the consonant skeleton survives all three."""
    return _MARKS.sub("", text)


def _windows(text: str, size: int = 5) -> set[str]:
    flat = "".join(text.split())
    return {flat[i:i + size] for i in range(max(0, len(flat) - size + 1))}


_GENERIC_SEARCH_PHRASES = (
    "มีทุนอะไรบ้าง", "มีทุนอะไร", "มีทุน", "ทุนสำหรับ", "สำหรับ", "ทำงานวิจัย", "ทำวิจัย",
    "งานวิจัย", "การวิจัย", "วิจัย", "อยากทำ", "อยาก", "มีไหม", "ได้ไหม", "หรือไม่",
    "หรือเปล่า", "บ้าง", "ครับ", "ค่ะ", "คะ", "ทุน",
)


def _meaningful_search_text(question: str) -> str:
    """Remove words shared by nearly every fund before measuring subject overlap."""
    text = question.lower()
    for phrase in sorted(_GENERIC_SEARCH_PHRASES, key=len, reverse=True):
        text = text.replace(phrase, " ")
    return " ".join(text.split())


def relevant_fund_clauses(question: str, chunks: list[dict], floor: int = 3) -> list[dict]:
    """Fund clauses that share real wording with the question.

    Retrieval always returns its best five passages, even when none of them is about what was asked.
    Suggesting a fund off a passage that merely ranked highest sends the reader somewhere wrong, which
    is worse than admitting the system does not know. Measured on this corpus the separation is clean:
    a question the documents cover shares eight or nine five-character windows with the right clause,
    while "อยากได้ไปญี่ปุ่น" shares none with anything, because no announcement names a country.
    """
    asked = _windows(_meaningful_search_text(question))
    if not asked:
        return []
    scored = [(len(asked & _windows(c["text"])), c) for c in chunks if FUND_CLAUSE.match(c.get("clause", ""))]
    return [c for score, c in sorted(scored, key=lambda pair: pair[0], reverse=True) if score >= floor]


def funds_mentioned_in(chunks: list[dict], head: int = 10) -> list[str]:
    """Funds named in the retrieved passages, so a question describing a need gets candidates.

    An annex clause names its fund through the reviewed rows (funds_in_clause). Matching by name misled
    there: five funds begin "ค่าตอบแทนผลงาน", so the first letters tied clause 2.5 to all five, and a list
    cut at four dropped the one the clause is about. Other passages are still matched by name, using the
    first few consonants, because the documents do not spell a fund the same way twice: the annex says
    "เงินรางวัลการตีพิมพ์เผยแพร่ผลงานวิจัย" where our row says "เงินรางวัลการตีพิมพ์ ผู้แต่งชื่อแรก".
    """
    found: list[str] = []
    by_name = []
    for chunk in chunks:
        reviewed = funds_in_clause(chunk)
        if reviewed is None:
            by_name.append(chunk)
        else:
            found += [fund for fund in reviewed if fund not in found]
    if by_name:
        blob = _skeleton(" ".join(c["text"] for c in by_name))
        found += [f for f in view().FUNDS if f not in found and _skeleton(f)[:head] and _skeleton(f)[:head] in blob]
    return found


def rules_for(fund: str | None, fiscal_year: int | None) -> list[dict]:
    """A rule with fiscal_year None comes from the central criteria and applies to every year."""
    return [r for r in view().RULES if r["fund"] == fund
            and (r["fiscal_year"] is None or not fiscal_year or r["fiscal_year"] == fiscal_year)
            and _document_reviewed(r["source"]["doc"])]


def rules_waiting(fund: str | None) -> bool:
    """A rule for the fund exists but comes from a scan nobody has approved yet."""
    return any(r["fund"] == fund and not _document_reviewed(r["source"]["doc"]) for r in view().RULES)


def _document_reviewed(doc: str) -> bool:
    """Rules derived from scans stay unavailable until a person approves that document's OCR."""
    chunks = [c for c in view().INGESTED_CHUNKS if c["doc"] == doc]
    return bool(chunks) and all(c["status"] == "active" and is_verified(c) for c in chunks)


def rrf(ranked_lists: list[list[dict]], k: int = 60, top_k: int = 5,
        weights: tuple[float, ...] | None = None) -> list[dict]:
    """Reciprocal rank fusion: merge lists that have incomparable scores."""
    if weights is not None and (len(weights) != len(ranked_lists) or any(w <= 0 for w in weights)):
        raise ValueError("RRF weights must match the rankings and be positive")
    score: dict[str, float] = {}
    seen: dict[str, dict] = {}
    for weight, hits in zip(weights or (1.0,) * len(ranked_lists), ranked_lists):
        for rank, hit in enumerate(hits, start=1):
            score[hit["id"]] = score.get(hit["id"], 0.0) + weight / (k + rank)
            seen[hit["id"]] = hit
    best = sorted(score, key=lambda i: score[i], reverse=True)[:top_k]
    return [seen[i] for i in best]


def semantic_guarded_rrf(
    keyword: list[dict],
    semantic: list[dict],
    hierarchy: list[dict],
    k: int = 60,
    top_k: int = 5,
    guard_k: int = 3,
) -> list[dict]:
    """Keep the top semantic vector matches first and fill remaining slots with RRF."""
    head = list(semantic[:guard_k])
    seen_ids = {hit["id"] for hit in head}
    fused = rrf([keyword, semantic, hierarchy], k=k, top_k=top_k + len(head))
    tail = [hit for hit in fused if hit["id"] not in seen_ids]
    return (head + tail)[:top_k]


# The legacy facade keeps the existing renderer and fixture tests usable. Runtime
# requests bind a PostgreSQL snapshot through the same public interface.
def view():
    import sys
    from app.modules.knowledge.context import snapshot
    return snapshot.get() or sys.modules[__name__]


def make_corpus(facts, chunks, rules, personas, aliases, revision=0):
    from types import SimpleNamespace
    from app.modules.knowledge.context import use_corpus
    funds = sorted({f["fund"] for f in facts}, key=len, reverse=True)
    vocab = set(TERMS) | set(TIERS) | set(funds)
    tokens = [tokenize(c["text"], vocab) for c in chunks]
    corpus = SimpleNamespace(
        FACTS=facts, CHUNKS=chunks, HAND_CHUNKS=[], INGESTED_CHUNKS=chunks,
        PUBLISHED_CHUNKS=[], RULES=rules, PERSONAS=personas, FUNDS=funds, VOCAB=vocab,
        revision=revision,
        ALIASES=sorted(((a.lower(), f) for f, names in aliases.get("aliases", {}).items() for a in names),
                       key=lambda p: len(p[0]), reverse=True),
        AMBIGUOUS=sorted(((a.lower(), fs) for a, fs in aliases.get("ambiguous", {}).items()),
                         key=lambda p: len(p[0]), reverse=True),
        _SEARCH_SNAPSHOT=(chunks, BM25Okapi(tokens) if tokens else None,
                          [Counter(trigrams(c["text"])) for c in chunks]))
    with use_corpus(corpus):
        corpus.DEFINITIONS = _build_definitions()
        corpus.ANNEX_FUNDS, corpus.ANNEX_PAGES = _annex_index()
        corpus._ANNEX_INDEX = (corpus.ANNEX_FUNDS, corpus.ANNEX_PAGES)
    return corpus


def _fixture_corpus():
    hand = _load("chunks.json")
    ingested = _load("chunks_ingested.json", [])
    published = [r for r in _load("chunks_published.json", []) if
                 r.get("status") == "active" and r.get("verified") is True and r.get("published") is True]
    corpus = make_corpus(_load("facts.json"), hand + ingested + published, _load("rules.json"),
                         {p["id"]: p for p in _load("personas.json")}, _load("aliases.json"))
    corpus.HAND_CHUNKS, corpus.INGESTED_CHUNKS, corpus.PUBLISHED_CHUNKS = hand, ingested, published
    return corpus


def __getattr__(name):
    if name in {"FACTS", "HAND_CHUNKS", "INGESTED_CHUNKS", "PUBLISHED_CHUNKS", "CHUNKS",
                "RULES", "PERSONAS", "FUNDS", "VOCAB", "DEFINITIONS", "ANNEX_FUNDS", "ANNEX_PAGES",
                "_ANNEX_INDEX", "_SEARCH_SNAPSHOT", "ALIASES", "AMBIGUOUS"}:
        with _DERIVED_INDEX_LOCK:
            if name not in globals():
                globals().update(vars(_fixture_corpus()))
        return globals()[name]
    raise AttributeError(name)
