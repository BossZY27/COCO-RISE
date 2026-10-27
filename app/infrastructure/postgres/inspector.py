"""Relational SQL database (data/coco_rise.db / PostgreSQL) + ChromaDB Vector Store (data/chroma_db).

Aligns the running system with the 4-layer dataset in IRFA-สถาปัตยกรรมระบบ.png and
Section 4.4 / 5.2 of the thesis proposal:
  - Layer 1 (PostgreSQL): document_registry   (ทะเบียนเอกสารต้นฉบับ Metadata)
  - Layer 2 (PostgreSQL + ChromaDB):
      document_chunks (ข้อความที่ตรวจแล้ว + BM25) + ChromaDB collection 'coco_rise_chunks' (bge-m3 1,024 มิติ)
  - Layer 3 (PostgreSQL): fund_facts          (ข้อเท็จจริงวงเงินแยกประเภท ceiling vs allocated)
                          eligibility_rules   (กฎคุณสมบัติและเงื่อนไขข้ามทุน)
  - Layer 4 (PostgreSQL): eval_dataset        (ชุดคำถามทดสอบและคำตอบอ้างอิง)
  - Profiles (PostgreSQL): applicant_profiles (โปรไฟล์ผู้ขอทุน ปกปิดข้อมูลส่วนบุคคลตามหลัก PDPA)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from app.infrastructure.chroma import vectors as chroma_vectors
from app.modules.knowledge import corpus as store

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / "data" / "coco_rise.db"
SCHEMA_SQL_PATH = Path(__file__).resolve().parent / "postgres_schema.sql"
_DB_LOCK = threading.Lock()
_SYNCED = False

ALLOWED_TABLES = (
    "document_registry",
    "document_chunks",
    "fund_facts",
    "eligibility_rules",
    "eval_dataset",
    "applicant_profiles",
)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _load_embedding_vectors() -> tuple[str, dict[str, list[float]]]:
    embed_model = "bge-m3:latest"
    embed_vectors: dict[str, list[float]] = {}
    if store.EMBEDDINGS.exists():
        try:
            raw_emb = json.loads(store.EMBEDDINGS.read_text(encoding="utf-8"))
            if isinstance(raw_emb, dict):
                embed_model = str(raw_emb.get("model") or embed_model)
                items = raw_emb.get("items")
                if isinstance(items, dict):
                    for cid, entry in items.items():
                        if isinstance(entry, dict) and isinstance(entry.get("vec"), list):
                            embed_vectors[cid] = entry["vec"]
                elif isinstance(raw_emb.get("vectors"), dict):
                    embed_vectors = raw_emb["vectors"]
        except Exception:
            embed_vectors = {}
    return embed_model, embed_vectors


def sync_database(force: bool = False) -> Path:
    """Create and populate data/coco_rise.db and sync vectors to ChromaDB (data/chroma_db)."""
    global _SYNCED
    with _DB_LOCK:
        if _SYNCED and not force and DB_PATH.exists():
            return DB_PATH
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = _connect()
        try:
            cur = conn.cursor()
            cur.executescript(
                """
                DROP TABLE IF EXISTS document_chunks;
                CREATE TABLE IF NOT EXISTS document_registry (
                    doc_id TEXT PRIMARY KEY,
                    doc_name TEXT NOT NULL,
                    fiscal_year INTEGER NULL,
                    doc_type TEXT NOT NULL,
                    source_format TEXT NOT NULL,
                    effective_status TEXT NOT NULL,
                    verified INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    fact_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    doc TEXT NOT NULL,
                    fiscal_year INTEGER NULL,
                    section TEXT NULL,
                    clause TEXT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    verified INTEGER NOT NULL,
                    chroma_id TEXT NULL,
                    vector_model TEXT NULL,
                    vector_dim INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS fund_facts (
                    id TEXT PRIMARY KEY,
                    fund TEXT NOT NULL,
                    tier TEXT NULL,
                    fiscal_year INTEGER NULL,
                    amount_type TEXT NOT NULL,
                    amount INTEGER NULL,
                    total_cap INTEGER NULL,
                    max_grants_per_year INTEGER NULL,
                    max_grants_per_person INTEGER NULL,
                    note TEXT NULL,
                    source_doc TEXT NOT NULL,
                    source_clause TEXT NULL,
                    source_page INTEGER NULL,
                    verified INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS eligibility_rules (
                    id TEXT PRIMARY KEY,
                    fund TEXT NOT NULL,
                    fiscal_year INTEGER NULL,
                    conditions_json TEXT NOT NULL,
                    source_doc TEXT NOT NULL,
                    source_clause TEXT NULL,
                    source_page INTEGER NULL,
                    verified INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS eval_dataset (
                    id TEXT PRIMARY KEY,
                    persona TEXT NOT NULL,
                    question TEXT NOT NULL,
                    expect_amount TEXT NULL,
                    expect_doc TEXT NULL,
                    expect_verdict TEXT NULL,
                    expect_status TEXT NULL
                );
                CREATE TABLE IF NOT EXISTS applicant_profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    staff_type_masked TEXT NOT NULL,
                    department_masked TEXT NOT NULL,
                    grants_summary_masked TEXT NOT NULL,
                    source TEXT NOT NULL,
                    as_of TEXT NULL
                );
                DELETE FROM document_registry;
                DELETE FROM document_chunks;
                DELETE FROM fund_facts;
                DELETE FROM eligibility_rules;
                DELETE FROM eval_dataset;
                DELETE FROM applicant_profiles;
                """
            )

            embed_model, embed_vectors = _load_embedding_vectors()
            if embed_vectors:
                try:
                    chroma_vectors.sync_corpus_chunks(list(store.CHUNKS), embed_vectors, embed_model)
                except Exception:
                    pass

            # 1. Populate document_chunks
            for chunk in store.CHUNKS:
                cid = str(chunk.get("id") or "")
                vec = embed_vectors.get(cid)
                v_dim = len(vec) if isinstance(vec, list) else 0
                cur.execute(
                    """
                    INSERT OR REPLACE INTO document_chunks
                    (id, doc, fiscal_year, section, clause, page, text, status, verified, chroma_id, vector_model, vector_dim)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cid,
                        str(chunk.get("doc") or ""),
                        chunk.get("fiscal_year"),
                        chunk.get("section"),
                        chunk.get("clause"),
                        int(chunk.get("page") or 1),
                        str(chunk.get("text") or ""),
                        str(chunk.get("status") or "active"),
                        1 if store.is_verified(chunk) else 0,
                        cid if v_dim > 0 else None,
                        embed_model if v_dim > 0 else None,
                        v_dim,
                    ),
                )

            # 2. Populate fund_facts
            for fact in store.FACTS:
                src = fact.get("source") or {}
                cur.execute(
                    """
                    INSERT OR REPLACE INTO fund_facts
                    (id, fund, tier, fiscal_year, amount_type, amount, total_cap,
                     max_grants_per_year, max_grants_per_person, note,
                     source_doc, source_clause, source_page, verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(fact.get("id") or ""),
                        str(fact.get("fund") or ""),
                        fact.get("tier"),
                        fact.get("fiscal_year"),
                        str(fact.get("amount_type") or "ceiling"),
                        fact.get("amount"),
                        fact.get("total_cap"),
                        fact.get("max_grants_per_year"),
                        fact.get("max_grants_per_person"),
                        fact.get("note"),
                        str(src.get("doc") or ""),
                        src.get("clause"),
                        src.get("page"),
                        1 if store.is_verified(fact) else 0,
                    ),
                )

            # 3. Populate eligibility_rules
            for rule in store.RULES:
                src = rule.get("source") or {}
                usable = bool(store.rules_for(rule.get("fund"), rule.get("fiscal_year")))
                cur.execute(
                    """
                    INSERT OR REPLACE INTO eligibility_rules
                    (id, fund, fiscal_year, conditions_json, source_doc, source_clause, source_page, verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(rule.get("id") or ""),
                        str(rule.get("fund") or ""),
                        rule.get("fiscal_year"),
                        json.dumps(rule.get("conditions") or [], ensure_ascii=False),
                        str(src.get("doc") or ""),
                        src.get("clause"),
                        src.get("page"),
                        1 if usable else 0,
                    ),
                )

            # 4. Populate document_registry (Layer 1: 4 announcements from the architecture diagram)
            docs_meta = [
                ("doc_01", "หลักเกณฑ์กลาง", None, "central_criteria", "digital_pdf", "active"),
                ("doc_02", "ประกาศ 41/2568", 2569, "annual_announcement", "scanned_ocr", "active"),
                ("doc_03", "ประกาศ 42/2568", 2569, "annual_announcement", "scanned_ocr", "active"),
                ("doc_04", "ประกาศ 10/2569", 2569, "annual_announcement", "scanned_ocr", "active"),
            ]
            for doc_id, doc_name, fyear, dtype, sformat, estatus in docs_meta:
                doc_chunks = [c for c in store.CHUNKS if doc_name in str(c.get("doc") or "")]
                doc_facts = [
                    f for f in store.FACTS if doc_name in str((f.get("source") or {}).get("doc") or "")
                ]
                is_ver = 1 if any(store.is_verified(c) for c in doc_chunks) else 0
                cur.execute(
                    """
                    INSERT OR REPLACE INTO document_registry
                    (doc_id, doc_name, fiscal_year, doc_type, source_format, effective_status, verified, chunk_count, fact_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        doc_name,
                        fyear,
                        dtype,
                        sformat,
                        estatus,
                        is_ver,
                        len(doc_chunks),
                        len(doc_facts),
                    ),
                )

            # 5. Populate eval_dataset (Layer 4)
            eval_path = store.DATA / "eval.json"
            eval_rows = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else []
            for idx, item in enumerate(eval_rows, start=1):
                cur.execute(
                    """
                    INSERT OR REPLACE INTO eval_dataset
                    (id, persona, question, expect_amount, expect_doc, expect_verdict, expect_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"ev{idx:02d}",
                        str(item.get("persona") or "P1"),
                        str(item.get("question") or ""),
                        item.get("expect_amount"),
                        item.get("expect_doc"),
                        item.get("expect_verdict"),
                        item.get("expect_status"),
                    ),
                )

            # 6. Populate applicant_profiles (with PDPA masking for display)
            for pid, p in store.PERSONAS.items():
                grants = p.get("grants_this_year") or {}
                cur.execute(
                    """
                    INSERT OR REPLACE INTO applicant_profiles
                    (id, name, staff_type_masked, department_masked, grants_summary_masked, source, as_of)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pid,
                        str(p.get("name") or pid),
                        "[ปกปิดตาม PDPA]" if p.get("staff_type") else "ไม่ระบุ (โหมดสาธารณะ)",
                        "[ปกปิดตาม PDPA]" if p.get("department") else "ไม่ระบุ (โหมดสาธารณะ)",
                        f"[ปกปิดตาม PDPA: มีประวัติ {len(grants)} รายการ]" if grants else "ไม่มีประวัติปีนี้ (0 รายการ)",
                        str(p.get("source") or "unknown"),
                        p.get("as_of"),
                    ),
                )

            conn.commit()
            _SYNCED = True
        finally:
            conn.close()
        return DB_PATH


def get_overview(table: str = "document_registry", only_verified: bool = False, q: str = "") -> dict:
    """Return database summary counts, ChromaDB stats, rows for the selected table, and PostgreSQL DDL."""
    sync_database()
    chosen = table if table in ALLOWED_TABLES else "document_registry"
    conn = _connect()
    try:
        cur = conn.cursor()
        counts = {}
        for tbl in ALLOWED_TABLES:
            counts[tbl] = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        verified_chunks = cur.execute("SELECT COUNT(*) FROM document_chunks WHERE verified = 1").fetchone()[0]
        verified_facts = cur.execute("SELECT COUNT(*) FROM fund_facts WHERE verified = 1").fetchone()[0]
        vector_chunks = cur.execute("SELECT COUNT(*) FROM document_chunks WHERE vector_dim > 0").fetchone()[0]

        clauses = []
        params: list = []
        if only_verified and chosen in ("document_registry", "document_chunks", "fund_facts", "eligibility_rules"):
            clauses.append("verified = 1")
        if q.strip():
            kw = f"%{q.strip()}%"
            if chosen == "document_chunks":
                clauses.append("(id LIKE ? OR doc LIKE ? OR clause LIKE ? OR text LIKE ?)")
                params.extend([kw, kw, kw, kw])
            elif chosen == "fund_facts":
                clauses.append("(id LIKE ? OR fund LIKE ? OR note LIKE ? OR source_doc LIKE ?)")
                params.extend([kw, kw, kw, kw])
            elif chosen == "eligibility_rules":
                clauses.append("(id LIKE ? OR fund LIKE ? OR conditions_json LIKE ?)")
                params.extend([kw, kw, kw])
            elif chosen == "eval_dataset":
                clauses.append("(id LIKE ? OR question LIKE ? OR expect_doc LIKE ?)")
                params.extend([kw, kw, kw])
            elif chosen == "applicant_profiles":
                clauses.append("(id LIKE ? OR name LIKE ?)")
                params.extend([kw, kw])
            else:
                clauses.append("(doc_id LIKE ? OR doc_name LIKE ?)")
                params.extend([kw, kw])

        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = [
            dict(r)
            for r in cur.execute(f"SELECT * FROM {chosen}{where_sql} LIMIT 200", params).fetchall()
        ]
    finally:
        conn.close()

    chroma_info = chroma_vectors.collection_stats()
    pg_schema = SCHEMA_SQL_PATH.read_text(encoding="utf-8") if SCHEMA_SQL_PATH.exists() else ""
    return {
        "engine": "PostgreSQL / SQLite (data/coco_rise.db) + ChromaDB (data/chroma_db)",
        "db_file": "data/coco_rise.db",
        "vector_db": chroma_info,
        "selected_table": chosen,
        "counts": counts,
        "stats": {
            "verified_chunks": verified_chunks,
            "verified_facts": verified_facts,
            "vector_chunks": chroma_info["vector_count"] or vector_chunks,
            "vector_model": "bge-m3:latest (1024 มิติ · ChromaDB)",
            "chroma_collection": chroma_info["collection"],
        },
        "rows": rows,
        "postgres_schema": pg_schema,
    }
