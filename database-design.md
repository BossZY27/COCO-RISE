# COCO RISE — Database Design

วันที่: 19 ตุลาคม 2569  
สถานะ: แบบร่างสำหรับออกแบบและหารือกับทีม ยังไม่ได้สร้างฐานข้อมูลหรือ migration

## 1. ขอบเขตและโครงสร้างฐานข้อมูล

ออกแบบการจัดเก็บเป็น 3 ส่วนตามหน้าที่ โดยมี 2 ส่วนภายในระบบและ 1 แหล่งข้อมูลภายนอก

| ส่วน | หน้าที่ | การติดตั้งที่เสนอ |
|---|---|---|
| Internal: PostgreSQL Structured KB | ข้อมูลทุน กฎ วันที่ เอกสาร หลักฐาน และข้อมูลที่นำเข้าจากวิทยาลัย | PostgreSQL database `coco_rise`, schema `core`, `knowledge`, `integration`, `audit` |
| Internal: pgvector Vector Store | Embeddings และการค้นหาเชิงความหมาย | extension `vector` ใน database เดียวกัน, schema `retrieval` |
| External: College DB | ข้อมูลกองทุนและประวัติการรับทุนที่วิทยาลัยดูแล | ระบบภายนอก อ่านผ่านช่องทางที่เจ้าของข้อมูลอนุญาต |

pgvector เป็น extension ของ PostgreSQL ไม่ใช่ database engine อีกตัว การแสดงเป็น 3 กล่องใน architecture หมายถึง 3 ส่วนเชิงหน้าที่ การใช้ database ภายในเดียวกันทำให้ใช้ foreign key, JOIN และ transaction ระหว่าง chunks กับ embeddings ได้

College DB เป็นแหล่งข้อมูลที่จะใช้ตามขอบเขตที่ทีมระบุ ส่วนชนิดฐานข้อมูล schema ช่องทางเข้าถึง และรายการคอลัมน์ยังต้องยืนยัน จึงไม่ออกแบบโดยสมมติว่าต้นทางเป็น PostgreSQL

ไฟล์ PDF/HTML ต้นฉบับเก็บใน file/object storage ส่วน database เก็บตำแหน่งไฟล์ hash และ metadata ข้อความที่สกัดได้เก็บใน PostgreSQL แบบนี้ไม่ได้รวมไฟล์ binary ทุกชนิดไว้ใน database

## 2. หลักการออกแบบ

- แยกประเภททุน (`grant_programs`) ออกจากรอบประกาศ (`grant_calls`) เช่น ทุนประเภทเดิมเปิดรับปี 2569 และ 2570 เป็นคนละรอบ
- เอกสารหนึ่งฉบับใช้กับหลายทุนได้ และทุนหนึ่งมีหลายเอกสารได้
- เก็บเวอร์ชันเอกสารแบบคงประวัติ chunks ต้องอ้างถึงเวอร์ชันที่ใช้สร้างจริง
- กฎ วันสำคัญ และข้อมูลที่สกัดต้องมีหลักฐานย้อนกลับถึงเอกสาร หน้า หรือระเบียนต้นทาง
- ข้อมูลจากการสกัดเริ่มต้นเป็น `draft` และเผยแพร่เมื่อผ่านการตรวจสอบ ไม่ถือว่า confidence ของโมเดลเป็นการยืนยันความถูกต้อง
- เก็บวันที่เป็น ค.ศ. ในชนิด `date` และแปลง พ.ศ. ตอนแสดงผล; ปีงบประมาณใช้ `fiscal_year_be` เพื่อระบุหน่วยให้ชัดเจน
- จำนวนเงินใช้ `numeric(14,2)` พร้อมสกุลเงิน ไม่ใช้ floating point
- Primary key ภายในใช้ UUID; รหัสจาก College DB ใช้ `text` และแยกจาก primary key ภายใน

## 3. PostgreSQL: Structured KB

คอลัมน์ในตารางต่อไปนี้เป็นรายการสำคัญ ไม่ใช่ DDL สมบูรณ์ ตารางข้อมูลที่แก้ไขได้มี `created_at`, `updated_at` แบบ `timestamptz` และสถานะที่กำหนดค่าที่อนุญาตไว้

### 3.1 ข้อมูลทุน

| ตาราง | ข้อมูลที่เก็บ | Key / ความสัมพันธ์ |
|---|---|---|
| `core.organizations` | รหัส ชื่อหน่วยงาน ประเภท และหน่วยงานแม่ | PK `organization_id`; FK `parent_id` → organizations |
| `core.grant_programs` | รหัสประเภททุน ชื่อ คำอธิบาย หน่วยงานผู้ให้ทุน | PK `program_id`; UNIQUE `program_code`; FK `funder_id` → organizations |
| `core.grant_calls` | รอบประกาศ ปีงบประมาณ ชื่อรอบ งบขั้นต่ำ/สูงสุด สกุลเงิน สถานะเผยแพร่ | PK `call_id`; UNIQUE `call_code`; FK `program_id` → grant_programs |
| `core.eligibility_rule_sets` | ชุดกฎ ระดับประเภททุนหรือรอบทุน เวอร์ชัน ช่วงวันที่มีผล สถานะตรวจสอบ | PK `rule_set_id`; FK `program_id` หรือ `call_id` โดยต้องมีเพียงหนึ่งค่า |
| `core.eligibility_rules` | เงื่อนไขข้อความ attribute/operator/value ที่ตรวจได้ `condition_json` และกลุ่ม AND/OR | PK `rule_id`; FK `rule_set_id` → eligibility_rule_sets |
| `core.timeline_events` | ประเภทเหตุการณ์ เช่น เปิดรับ ปิดรับ ประกาศผล วันที่/เวลาถ้าระบุ คำอธิบาย | PK `event_id`; FK `call_id` → grant_calls |

`timeline_events` เป็นแหล่งวันที่ของรอบทุน ไม่เก็บ open/close ซ้ำใน grant_calls ในระยะแรก หากจำเป็นสามารถทำ view สำหรับแสดงวันเปิดปิดได้ กรณีเอกสารระบุเพียงวันที่ให้เก็บ `event_date`; เวลา `event_at` เป็น optional และต้องสอดคล้องกับวันที่ในเขต Asia/Bangkok

กฎที่ยังแปลงเป็นเงื่อนไขเชิงเครื่องไม่ได้ให้เก็บข้อความและ `evaluation_mode = manual` ผลตรวจคุณสมบัติต้องรองรับ `unknown` เมื่อข้อมูลผู้สมัครหรือประวัติไม่ครบ กฎทั่วไปกับกฎเฉพาะรอบต้องมีลำดับการบังคับใช้ที่ตรวจสอบแล้ว ไม่ใช้หลักว่าเอกสารใหม่กว่าชนะเสมอ

### 3.2 เอกสารและหลักฐาน

| ตาราง | ข้อมูลที่เก็บ | Key / ความสัมพันธ์ |
|---|---|---|
| `knowledge.documents` | ชื่อ ประเภท PDF/HTML/form หน่วยงานผู้ออก หมายเลขประกาศ URL ต้นทาง | PK `document_id`; FK `issuer_id` → organizations |
| `knowledge.document_versions` | ลำดับเวอร์ชัน storage URI, SHA-256, MIME, วันที่ประกาศ/มีผล สถานะ active/superseded/withdrawn | PK `version_id`; FK `document_id`; UNIQUE `(document_id, version_no)` |
| `knowledge.grant_documents` | บทบาทเอกสาร เช่น หลักเกณฑ์ ประกาศรอบ หรือเอกสารแก้ไข | FK `version_id`; FK `program_id` หรือ `call_id` เพียงหนึ่งค่า; unique ต่อเอกสาร-เป้าหมาย-บทบาท |
| `knowledge.document_pages` | เลขหน้า PDF ข้อความดิบ ข้อความ normalized วิธี text/OCR และ page label ที่พิมพ์ในเอกสาร | PK `page_id`; FK `version_id`; UNIQUE `(version_id, page_number)` |
| `knowledge.chunk_sets` | การแบ่งข้อความแต่ละครั้ง chunker/version/config และ ingestion run | PK `chunk_set_id`; FK `version_id`, `run_id` |
| `knowledge.document_chunks` | ข้อความ chunk ลำดับ section hash สถานะเผยแพร่ | PK `chunk_id`; FK `chunk_set_id`; UNIQUE `(chunk_set_id, ordinal)` |
| `knowledge.chunk_pages` | ช่วงข้อความของ chunk ในแต่ละหน้า รองรับ chunk ข้ามหน้า | PK `(chunk_id, page_id)`; FK → chunks/pages; optional offsets |
| `knowledge.evidence_records` | หลักฐานแบบ `document` หรือ `college_record`; สำหรับเอกสารระบุ version/page/chunk/quote; สำหรับ College ระบุ source record และ field path | PK `evidence_id`; FK ไปยังเอกสารหรือ source_records ตามชนิดหลักฐาน |
| `knowledge.rule_evidence` | กฎกับหลักฐานหนึ่งหรือหลายรายการ | PK `(rule_id, evidence_id)`; FK ทั้งสองฝั่ง |
| `knowledge.event_evidence` | กำหนดเวลากับหลักฐาน | PK `(event_id, evidence_id)`; FK ทั้งสองฝั่ง |
| `knowledge.call_evidence` | หลักฐานของข้อมูลรอบทุน เช่น ชื่อ วงเงิน และปีงบประมาณ | PK `(call_id, field_name, evidence_id)`; FK ทั้งสองฝั่ง |

เก็บไฟล์เวอร์ชันเก่าและหลักฐานเดิมไว้เพื่อให้ citation ย้อนหลังยังตรวจได้ หน้า PDF นับจาก 1 ส่วน page label อาจเป็นเลขไทย/เลขพิมพ์ที่ต่างกัน HTML/form อาจไม่มีเลขหน้า ให้เก็บ locator เช่น heading/anchor ใน evidence_records แทน ไม่สร้างเลขหน้าที่ไม่มีจริง

chunk_pages ต้องอ้างถึงหน้าของ document version เดียวกับ chunk_set บังคับด้วย composite FK หรือ constraint trigger เมื่อทำ DDL และไม่ใช้ CASCADE ลบเอกสารที่มีหลักฐานอ้างอิงอยู่

### 3.3 ข้อมูลจาก College DB

ชื่อและฟิลด์ต่อไปนี้เป็นแบบจำลองภายในที่เสนอ ไม่ใช่ schema ที่ตรวจพบจาก College DB

| ตาราง | ข้อมูลที่เก็บ | Key / ความสัมพันธ์ |
|---|---|---|
| `integration.source_systems` | ชื่อระบบ วิธีเข้าถึง และ owner reference; ไม่เก็บรหัสผ่าน | PK `source_system_id` |
| `integration.sync_runs` | เวลาเริ่ม/จบ สถานะ จำนวนอ่าน/เพิ่ม/แก้ไข/ผิดพลาด cursor และเวลาข้อมูลล่าสุด | PK `sync_run_id`; FK `source_system_id` |
| `integration.source_records` | สำเนาเฉพาะฟิลด์ที่ได้รับอนุญาต รหัสตาราง/ระเบียนต้นทาง payload JSONB, hash, source_updated_at, observed_at และสถานะ | PK `source_record_id`; FK `source_system_id`, `sync_run_id`; unique ตาม source/type/key/version hash |
| `integration.entity_mappings` | map รหัสต้นทางเข้ากับ organization/program/call/researcher/award ภายใน | PK `mapping_id`; FK `source_system_id` และ FK เป้าหมายแบบระบุชนิด; UNIQUE `(source_system_id, entity_type, external_key)` |
| `core.researchers` | รหัสอ้างอิง ชื่อที่อนุญาตให้ใช้ สังกัด และข้อมูลที่จำเป็นต่อการตรวจคุณสมบัติ | PK `researcher_id`; FK `organization_id` |
| `core.funding_awards` | รายการได้รับทุน วงเงินอนุมัติ วันที่ สถานะ และหลักฐานจากต้นทาง | PK `award_id`; FK `program_id`, optional `call_id`, `source_record_id` |
| `core.award_recipients` | ผู้รับทุนหรือสมาชิกทีมและบทบาท | PK `(award_id, researcher_id)`; FK ทั้งสองฝั่ง |

entity_mappings ใช้ target FK แยกชนิดและ CHECK ให้มีเป้าหมายที่สอดคล้องเพียงหนึ่งค่า ไม่ใช้ generic target UUID ที่ไม่สามารถตรวจ referential integrity ได้

funding_awards อาจรู้เพียงประเภททุนแต่ยังจับคู่รอบไม่ได้ จึงอนุญาต call_id เป็น NULL และต้องตรวจว่าหากระบุ call_id จะอยู่ใน program_id เดียวกัน การไม่มีรายการรับทุนไม่เท่ากับยืนยันว่าไม่เคยรับ ต้องตรวจความครบถ้วนและช่วงเวลาที่ sync ด้วย

## 4. pgvector: Vector Store

| ตาราง | ข้อมูลที่เก็บ | Key / ความสัมพันธ์ |
|---|---|---|
| `retrieval.embedding_models` | ชื่อโมเดล revision/provider จำนวนมิติ distance metric และ preprocessing config | PK `model_id`; UNIQUE `(name, revision, config_hash)` |
| `retrieval.embedding_indexes` | ชุดดัชนีที่ใช้งาน โมเดล chunking config สถานะ building/active/retired | PK `index_id`; FK `model_id` |
| `retrieval.chunk_embeddings` | vector เวลาสร้าง และข้อมูลตรวจสอบความสำเร็จ | PK `(index_id, chunk_id)`; FK `index_id`, `chunk_id` |

ข้อความต้นฉบับอยู่ใน knowledge.document_chunks และเชื่อมด้วย chunk_id ไม่ต้องคัดลอกข้อความซ้ำลงตาราง vector

ยังไม่กำหนดจำนวนมิติ: เมื่อทีมยืนยันโมเดลจึงเลือก `vector(d)` หรือชนิดที่เหมาะกับมิติและ index หากรองรับหลายโมเดลให้แยก partition/ตารางและ index ตาม embedding space ไม่เปรียบเทียบ vector ข้ามโมเดล แม้จำนวนมิติเท่ากัน ก่อนเปลี่ยนชุด active ต้องตรวจจำนวน chunks และความครบถ้วนของ embeddings

## 5. ความสัมพันธ์หลัก

| ต้นทาง | ความสัมพันธ์ | ปลายทาง |
|---|---|---|
| organization | 1:N | grant_programs / researchers |
| grant_program | 1:N | grant_calls / funding_awards |
| grant_call | 1:N | timeline_events / funding_awards |
| program หรือ call | 1:N | eligibility_rule_sets → eligibility_rules |
| program/call | M:N ผ่าน grant_documents | document_versions |
| document | 1:N | document_versions |
| document_version | 1:N | document_pages / chunk_sets |
| chunk_set | 1:N | document_chunks |
| chunk | M:N ผ่าน chunk_pages | document_pages |
| chunk | 1:N | chunk_embeddings ตาม index_id |
| embedding_model | 1:N | embedding_indexes → chunk_embeddings |
| rule/event/call field | M:N ผ่านตาราง evidence | evidence_records |
| funding_award | M:N ผ่าน award_recipients | researchers |
| College DB | read-only import | source_records → entity_mappings และ core tables |

ไม่มี foreign key ข้ามไป College DB โดยตรง ความสัมพันธ์ต้นทางเก็บเป็น source_system/type/external_key และตรวจสอบตอน sync

## 6. แผน Index

PK และ UNIQUE สร้าง index ให้อยู่แล้ว ไม่สร้างซ้ำ Index ด้านล่างเป็นแบบเสนอ ต้องตรวจด้วย query จริงและ EXPLAIN ก่อนใช้ทุกตัว

| ตาราง / คอลัมน์ | ชนิด | Query ที่รองรับ |
|---|---|---|
| grant_programs `(funder_id)` | B-tree | ทุนของหน่วยงาน |
| grant_calls `(program_id, fiscal_year_be)` | B-tree | รอบทุนของประเภทหนึ่งในแต่ละปี |
| grant_calls `(fiscal_year_be)` WHERE publication_status = 'published' | Partial B-tree | รายการทุนที่เผยแพร่ในปีที่เลือก |
| timeline_events `(call_id, event_type, event_date)` | B-tree | วันสำคัญของรอบทุน |
| timeline_events `(event_type, event_date)` | B-tree | รอบที่กำลังเปิดรับ/ใกล้ปิด โดยใช้วันที่เป็น query parameter |
| eligibility_rule_sets `(program_id, valid_from)`, `(call_id, valid_from)` | Partial B-tree สำหรับ FK ที่ไม่ NULL | กฎที่ใช้กับประเภท/รอบทุน |
| eligibility_rules `(rule_set_id)` | B-tree | อ่านกฎทั้งชุด |
| grant_documents `(program_id, version_id)`, `(call_id, version_id)` | Partial UNIQUE เมื่อ FK ไม่ NULL เพิ่ม role ใน unique key | ป้องกัน link ซ้ำและอ่านเอกสารของทุน |
| grant_documents `(version_id)` | B-tree | ย้อนจากเอกสารไปหาทุน |
| document_versions `(document_id, version_no)` | UNIQUE B-tree | ประวัติเวอร์ชัน |
| document_versions `(document_id)` WHERE status = 'active' | Partial UNIQUE | active version สูงสุดหนึ่งรายการต่อ document |
| document_versions `(content_sha256)` | B-tree, ไม่บังคับ unique ทั้งระบบ | ตรวจไฟล์ซ้ำโดยไม่รวมเอกสารคนละ identity |
| document_pages `(version_id, page_number)` | UNIQUE B-tree | เปิดหน้าต้นฉบับ |
| chunk_sets `(version_id)`; document_chunks `(chunk_set_id, ordinal)` | B-tree / UNIQUE | chunks ของเวอร์ชันและลำดับอ่าน |
| chunk_pages `(page_id, chunk_id)` | B-tree | ย้อนจากหน้าไปหา chunks; PK รองรับอีกทิศทาง |
| chunk_embeddings `(index_id, chunk_id)` | PK B-tree | ดึง embeddings ของชุดดัชนี |
| chunk_embeddings `(chunk_id)` | B-tree | ย้อนจาก chunk ไปทุก embeddings |
| embedding column แยกตามโมเดล/มิติ | เริ่ม exact search; เพิ่ม HNSW cosine เมื่อวัดแล้วจำเป็น | semantic nearest-neighbor search |
| funding_awards `(program_id, awarded_at)` และ `(call_id)` | B-tree | ประวัติทุนตามประเภท/รอบ |
| award_recipients `(researcher_id, award_id)` | B-tree | ประวัติการรับทุนรายคน; PK รองรับอีกทิศทาง |
| source_records `(source_system_id, entity_type, external_key, observed_at)` | B-tree | ย้อนดูข้อมูลต้นทางและเวอร์ชัน |
| sync_runs `(source_system_id, started_at)` | B-tree | สถานะและความสดใหม่ของการ sync |

เมื่อทำ DDL ให้ตรวจ FK ที่ใช้ JOIN ทั้งหมด รวมถึง issuer_id, organization parent_id, model_id, run_id, sync_run_id, source_record_id และ evidence FKs ว่ามี index โดยไม่ซ้ำ prefix ของ PK/UNIQUE ที่มีอยู่ ตาราง evidence ให้เพิ่ม index ฝั่ง evidence_id สำหรับการค้นย้อนกลับ

ไม่ใช้ `CURRENT_DATE`/`now()` ใน predicate ของ partial index สำหรับทุนที่ยังไม่หมดอายุ ให้ index วันที่แล้วส่งวันปัจจุบันใน query กรองข้อมูลแทน

Keyword search ภาษาไทยต้องเลือกวิธีตัดคำก่อนสร้าง tsvector/GIN และทดสอบกับคำถามจริง ไม่ถือว่า PostgreSQL default configuration ตัดคำไทยได้ตรงความต้องการ ส่วน GIN บน JSONB สร้างเฉพาะฟิลด์ที่มี query containment/path จริง ไม่สร้างทุก payload โดยอัตโนมัติ

## 7. เส้นทางการนำเข้าและค้นหา

1. Admin นำเข้า PDF/HTML → สร้าง document/version และ ingestion run → สกัดหน้าและแบ่ง chunk
2. สกัดประเภททุน รอบทุน กฎ และวันสำคัญ → บันทึก draft พร้อม evidence → Admin ตรวจและเผยแพร่
3. สร้าง embedding จาก chunks ที่เลือก → บันทึกตาม model/index → เปิดใช้เมื่อประมวลผลครบ
4. College DB → connector แบบ read-only → sync_runs/source_records → ตรวจและ map รหัส → core tables
5. Query service ตรวจขอบเขตการเข้าถึง → SQL lookup และ vector search → รวมหลักฐาน → LLM ตอบพร้อม citation

การตอบเรื่องประวัติทุนใช้ค่าจาก structured records และอ้าง source key/เวลาที่ sync ส่วนข้อความระเบียบอ้าง document version/หน้า/chunk แสดงข้อมูลส่วนบุคคลตามสิทธิ์ผู้ใช้ กรองสิทธิ์ก่อนส่งบริบทเข้า LLM

การเผยแพร่ข้อมูล structured และสถานะ vector ต้องตรวจร่วมกันใน transaction หรือใช้ ingestion status ป้องกัน query เห็นข้อมูลที่ประมวลผลครึ่งหนึ่ง

## 8. งานประมวลผลและตรวจสอบย้อนหลัง

เพิ่ม `audit.ingestion_runs` สำหรับ input version, ขั้นตอน, tool/config version, เวลา สถานะ และ error; chunk_sets อ้างถึง run นี้ เพิ่ม `audit.change_logs` สำหรับ entity/action/actor/เวลาและรายละเอียดที่จำเป็น โดยป้องกันไม่ให้ log เปิดเผยข้อมูลเกินสิทธิ์

เวอร์ชันที่ถูกแทนที่ไม่ปรากฏใน current retrieval แต่คง citation ย้อนหลังไว้ ข้อมูล College ที่ไม่พบในรอบ sync ใหม่ให้ยืนยันก่อนว่ารอบนั้นครอบคลุมครบ ไม่ลบหรือสรุปว่ายกเลิกจากการหายไปเพียงครั้งเดียว

## 9. เรื่องที่ต้องยืนยันก่อนทำ DDL

- College DB ใช้ระบบอะไร เปิดผ่าน API/read-only view/export หรือ direct connection และมี primary key/updated_at หรือไม่
- ข้อมูลที่เข้าถึงได้จริง: ประเภททุน รอบทุน ผู้รับทุน สมาชิกทีม วงเงิน สถานะ และช่วงปีย้อนหลัง
- วิธีจับคู่บุคลากรกับระบบเรา รวมถึงขอบเขตสิทธิ์การแสดงประวัติทุน
- โมเดล embedding, revision, มิติ และรูปแบบ preprocessing จากทีม Demo
- ตัวอย่างเอกสารแก้ไขและวิธีระบุว่ากฎใดแทนที่กฎเดิม; ห้ามตั้งอายุเอกสาร 1/5 ปีเองหากแหล่งข้อมูลไม่ได้ระบุ

## 10. แหล่งอ้างอิง

- [System Architecture ของโครงการ](Architecture.md)
- [Data Management Architecture เดิม](data-management-architecture.md) — ใช้เป็นข้อมูลประกอบ แบบร่างนี้ปรับการจัดเก็บภายในเป็น PostgreSQL + pgvector
- [pgvector: ชนิด vector, exact/approximate search และ filtering](https://github.com/pgvector/pgvector)
- [PostgreSQL: Constraints และ Foreign Keys](https://www.postgresql.org/docs/18/ddl-constraints.html)
- [PostgreSQL: Partial Indexes](https://www.postgresql.org/docs/18/indexes-partial.html)

การออกแบบนี้เป็นข้อเสนอที่อิงความต้องการโครงการ ยังไม่มี benchmark ด้วยข้อมูลจริงหรือการยืนยัน schema ของ College DB
