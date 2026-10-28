# RISE API

Swagger `/docs` และ `/openapi.json` เป็น contract หลัก API ทั้งหมดใช้ `/api/*` การเลือกโมเดลเป็นค่าของผู้ดูแล ไม่รับ model หรือ persona จาก client

| เส้นทาง | หน้าที่ |
| --- | --- |
| `POST /api/query` | question, history สูงสุด 8 ข้อ ข้อละ 400 ตัวอักษร และ context แบบเลือกส่ง |
| `POST /api/feedback` | request_id, helpful และ note; ต้องอ้างถึง query ที่มีอยู่ |
| `/api/status`, `/api/ready` | สถานะบริการและ readiness ของฐานข้อมูล |
| `/api/documents` | รายการเอกสารและอัปโหลด raw application/pdf |
| `/api/documents/inspect-pdf` | ตรวจตัวอย่างโดยไม่บันทึก |
| `/api/documents/{id}/versions` | รายการรุ่นและอัปโหลดรุ่นใหม่ |
| `/api/versions/{id}` | ข้อมูลรุ่น พร้อม pages/chunks, แก้หน้า, review, publish, withdraw และ source |
| `/api/sources/{version_id}/pages/{page}` | เปิด PDF ที่เผยแพร่แล้วตามรุ่นและหน้าอ้างอิง |
| `/api/jobs` | รายการงาน สถานะงาน และ retry |
| `/api/facts`, `/api/rules` | สร้าง/แก้ draft และส่งผล review แยกจากการอนุมัติ PDF |
| `/api/recent`, `/api/db/overview` | ประวัติ query และข้อมูล PostgreSQL สำหรับผู้ดูแล |

งานผู้ดูแลต้องส่ง `X-Admin-Token` ไม่มีโทเคนใน query string รูปแบบ PDF ใช้ header `X-PDF-Name` และ query parameters `title`, `fiscal_year_be` แบบเลือกส่ง ตาม Swagger

ผล query มี `request_id`, `status`, `answer`, `answer_source`, citations แบบ document/version/page/chunk, `verification`, `timings_ms`, `context` และ `processing` ซึ่งแยกการใช้ LLM ตีความ/เรียบเรียง วิธีค้น และเหตุผล fallback

context ใช้แก้คำอ้างอิงในบทสนทนา ไม่ใช่หลักฐาน ข้อความคำถามใหม่ชนะบริบทเดิม คำตอบที่บล็อกไม่คืน context เพื่อป้องกันการจดจำหัวข้อที่ตรวจไม่ผ่าน
