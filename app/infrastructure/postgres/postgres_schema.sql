-- =============================================================================
-- COCO-RISE (IRFA) Database Schema for PostgreSQL + ChromaDB Vector Store
-- PostgreSQL เก็บตารางข้อมูลเชิงสัมพันธ์ 4 ชั้น (Relational Tables)
-- ChromaDB (data/chroma_db · collection 'coco_rise_chunks') เก็บเวกเตอร์ bge-m3 (1,024 มิติ)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- ชั้นที่ 1: ทะเบียนเอกสารต้นฉบับ (Document Registry / Metadata)
-- เก็บข้อมูลประกาศทุน 4 ฉบับ เลขที่ ปีงบประมาณ ประเภท และสถานะการตรวจทาน
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_registry (
    doc_id          VARCHAR(64) PRIMARY KEY,
    doc_name        VARCHAR(255) NOT NULL UNIQUE,
    fiscal_year     INT NULL,
    doc_type        VARCHAR(64) NOT NULL, -- 'central_criteria' หรือ 'annual_announcement'
    source_format   VARCHAR(64) NOT NULL, -- 'digital_pdf' หรือ 'scanned_ocr'
    effective_status VARCHAR(32) NOT NULL DEFAULT 'active',
    verified        BOOLEAN NOT NULL DEFAULT FALSE,
    chunk_count     INT NOT NULL DEFAULT 0,
    fact_count      INT NOT NULL DEFAULT 0
);

-- -----------------------------------------------------------------------------
-- ชั้นที่ 2: ข้อความที่ตรวจแล้ว (document_chunks ใน PostgreSQL + เวกเตอร์ใน ChromaDB)
-- แบ่งตามข้อและแถวตาราง พร้อมเลขหน้า และเชื่อมกับ ChromaDB collection 'coco_rise_chunks'
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id              VARCHAR(64) PRIMARY KEY,
    doc             VARCHAR(255) NOT NULL,
    fiscal_year     INT NULL,
    section         VARCHAR(255) NULL,
    clause          VARCHAR(64) NULL,
    page            INT NOT NULL,
    text            TEXT NOT NULL,
    status          VARCHAR(32) NOT NULL DEFAULT 'active',
    verified        BOOLEAN NOT NULL DEFAULT FALSE,
    chroma_id       VARCHAR(64) NULL,     -- อ้างอิง ID ใน ChromaDB collection 'coco_rise_chunks'
    vector_model    VARCHAR(64) NULL DEFAULT 'bge-m3:latest',
    vector_dim      INT NOT NULL DEFAULT 1024
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_filter
    ON document_chunks (verified, status, fiscal_year);

-- -----------------------------------------------------------------------------
-- ชั้นที่ 3 (ส่วนที่ 1): ข้อเท็จจริงวงเงินแยกตามประเภท (facts)
-- แยกเพดานหลักเกณฑ์กลาง (ceiling) ออกจากวงเงินจัดสรรประจำปี (allocated)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fund_facts (
    id                      VARCHAR(64) PRIMARY KEY,
    fund                    VARCHAR(255) NOT NULL,
    tier                    VARCHAR(64) NULL,
    fiscal_year             INT NULL,
    amount_type             VARCHAR(32) NOT NULL CHECK (amount_type IN ('ceiling', 'allocated')),
    amount                  NUMERIC(12, 2) NULL,
    total_cap               NUMERIC(12, 2) NULL,
    max_grants_per_year     INT NULL,
    max_grants_per_person   INT NULL,
    note                    TEXT NULL,
    source_doc              VARCHAR(255) NOT NULL,
    source_clause           VARCHAR(64) NULL,
    source_page             INT NULL,
    verified                BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_fund_facts_lookup
    ON fund_facts (fund, fiscal_year, amount_type, verified);

-- -----------------------------------------------------------------------------
-- ชั้นที่ 3 (ส่วนที่ 2): กฎคุณสมบัติและเงื่อนไขข้ามทุน (rules)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eligibility_rules (
    id              VARCHAR(64) PRIMARY KEY,
    fund            VARCHAR(255) NOT NULL,
    fiscal_year     INT NULL,
    conditions      JSONB NOT NULL,
    source_doc      VARCHAR(255) NOT NULL,
    source_clause   VARCHAR(64) NULL,
    source_page     INT NULL,
    verified        BOOLEAN NOT NULL DEFAULT FALSE
);

-- -----------------------------------------------------------------------------
-- ชั้นที่ 4: ชุดคำถามทดสอบและคำตอบอ้างอิง (Evaluation Dataset)
-- สำหรับวัด Recall@k, MRR, Faithfulness, Citation Precision/Recall
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS eval_dataset (
    id                  VARCHAR(64) PRIMARY KEY,
    question            TEXT NOT NULL,
    expected_fund       VARCHAR(255) NULL,
    expected_intent     VARCHAR(64) NULL,
    expected_citations  JSONB NOT NULL DEFAULT '[]'::jsonb,
    expected_numbers    JSONB NOT NULL DEFAULT '[]'::jsonb
);

-- -----------------------------------------------------------------------------
-- ตารางผู้ใช้และโปรไฟล์ (users / profiles)
-- แยกเก็บออกจากโมเดล LLM ตามหลักการคุ้มครองข้อมูลส่วนบุคคล (PDPA)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS applicant_profiles (
    id                  VARCHAR(64) PRIMARY KEY,
    name                VARCHAR(255) NOT NULL,
    staff_type          VARCHAR(128) NULL,
    department          VARCHAR(255) NULL,
    grants_this_year    JSONB NOT NULL DEFAULT '{}'::jsonb,
    source              VARCHAR(64) NOT NULL DEFAULT 'demo',
    as_of               VARCHAR(32) NULL
);
