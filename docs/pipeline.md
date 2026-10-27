# RISE pipelines

## Query

```text
analyze → retrieve → compose → verify → answer or fallback
```

ระบบตรวจ input/rate limit แล้ววิเคราะห์ fund, tier, fiscal year และ intent แบบ deterministic จากนั้นค้น BM25 และ BGE-M3/pgvector แบบ exact cosine ก่อนรวมผลด้วย RRF Facts และ rules ที่อนุมัติถูกเลือกแยกจาก passage retrieval ระบบสร้าง verified template และอาจให้ Qwen3 สร้าง candidate แบบจำกัดขอบเขต Candidate ต้องผ่านการตรวจ number, year, fund, tier, qualification, note และ citation รวมถึง corpus revision check จึงส่งได้ หาก Ollama ล้มเหลวหรือ candidate ไม่ผ่าน ระบบใช้ verified template; หาก template ไม่ผ่านจะตอบ `blocked` โดยไม่ปล่อยข้อความที่ตรวจไม่ผ่าน `/v1/query/stream` ส่งเฉพาะ verified preview/done events

## Ingestion

```text
upload → extract/OCR → chunk → review → index → publish
```

PDF ถูกตรวจชนิด ขนาด จำนวนหน้า และมิติ ก่อนเก็บแบบ content-addressed ใน private storage Worker claim งานด้วย lease แล้ว extract text หรือ OCR ด้วย Tesseract `tha+eng` จากนั้นสร้าง draft pages/chunks Human reviewer แก้ข้อความและยืนยัน checksum กับทุกหน้า งาน indexing สร้าง generation ใหม่ให้เสร็จก่อน activation และ publication transaction จึงเปิดหลักฐานให้ current retrieval Facts/rules ต้องมี inline evidence และผ่าน decision แยกต่างหาก

Worker reclaim งานที่ lease หมดอายุได้ภายใน retry limit เจ้าของ lease เก่าถูก fence ไม่ให้ commit Heartbeat, extraction และ index activation ใช้ transaction สั้น Retry อาจแทนที่ machine-only extraction แต่ห้ามเขียนทับ human edits หรือ evidence ที่ structured knowledge อ้างอยู่

## Document update

```text
create version → review → prepare index → activate → supersede old version
```

เอกสารใหม่สร้าง immutable version ภายใต้ document เดิม หลัง review ระบบเตรียม index generation แล้ว activate version ใหม่พร้อมเปลี่ยน current pointer และ supersede version เดิมแบบ atomic การ withdraw เอา version ออกจาก current retrieval เช่นกัน PDF ของ historical version ยังเปิดผ่าน citation endpoint ที่ระบุ version/page/chunk ได้ แม้ version นั้นถูก supersede หรือ withdrawn
