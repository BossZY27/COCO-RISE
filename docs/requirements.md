# Requirements mapping

แหล่งข้อกำหนดหลักคือเค้าโครง `RISE-เค้าโครงโครงงานคอมพิวเตอร์-2569-v3.pdf` ส่วน `database-design.md` ใช้เป็นแบบร่างเพิ่มเติมสำหรับ schema เป้าหมาย การเลือก PostgreSQL + pgvector ในเอกสารนี้ขยายจากต้นแบบ JSON/in-memory ที่ระบุในรายงาน และยังไม่ใช่ระบบที่ติดตั้งแล้ว

| ข้อกำหนด v3 | องค์ประกอบ / API เป้าหมาย | สถานะและเกณฑ์ตรวจรับ |
|---|---|---|
| เตรียม PDF สแกนด้วย OCR และตรวจเทียบต้นฉบับ | ingestion worker, document pages/chunks; `/v1/admin/documents` และ jobs | ระบบนำเข้าและตรวจเอกสารมีในเดโม; repository/queue ที่ใช้ PostgreSQL และ production worker ต้องทำสำหรับ MVP; draft ต้องค้นไม่พบ |
| ถามตอบภาษาไทยพร้อมหน้าอ้างอิง | Query API, retrieval service, source/citation store | Query และ citation มีในเดโม; แสดงเวอร์ชันเอกสารและ locator ที่ย้อนกลับต้นฉบับได้ |
| Hybrid retrieval: BM25 + BGE-M3 + RRF | keyword และ vector retrievers, fusion service; structured facts/rules lookup แยกต่างหาก | BM25 และ fallback trigram มีในเดโม; vector ใช้ได้เมื่อ Ollama พร้อม; การค้น pgvector และ RRF เป้าหมายต้องพัฒนาสำหรับ MVP |
| เก็บ facts/rules ที่ตรวจแล้ว | structured knowledge repository, admin review APIs | JSON facts/rules มีในเดโม; API CRUD และ PostgreSQL schema ต้องพัฒนาสำหรับ MVP; หลักฐานไม่ผ่าน review ห้ามเข้าผลค้น |
| ให้ LLM เรียบเรียง แล้วตรวจคำตอบก่อนส่ง | Ollama answer client, answer verifier, deterministic fallback | เดโมปัจจุบันปิด free-form answer LLM; MVP ต้องพัฒนาและประเมิน verifier เพิ่มเติมก่อนเปิดเส้นทางนี้ |
| เปรียบเทียบ 3 วิธี | evaluation runner และชุดคำถามที่ผู้ตรวจยืนยัน | ต้องพัฒนาสำหรับการทดลอง; ห้ามอ้างผลก่อนกำหนด protocol และรันจริง |
| JSON ปัจจุบัน; PostgreSQL/pgvector เป็นทางขยายในรายงาน | MVP ใช้ PostgreSQL + pgvector ตามการตัดสินใจล่าสุด; PDF แยก file storage | ยังไม่มี migration หรือฐานข้อมูลนี้ในเดโม |
| จำกัดขอบเขต ไม่จับคู่ทุนหรือเช็คข้อเสนอ | API และ UI ไม่เสนอการจัดอันดับเฉพาะบุคคล | การจับคู่ทุน ตรวจข้อเสนอโครงการ และการยื่นแทนผู้ใช้เป็นส่วนต่อขยายที่ไม่อยู่ใน MVP |
| ยังไม่เชื่อมฐานข้อมูลบุคลากร | External integration boundary | ส่วนต่อขยายอนาคต; ใช้ข้อมูลที่ไม่ระบุตัวบุคคล/ข้อมูลสมมติในการทดลอง MVP |

## การประเมิน 3 วิธีตามรายงาน

ใช้คำถามและ gold evidence ชุดเดียวกันภายใต้การตั้งค่าที่บันทึกเวอร์ชันไว้:

1. Text RAG ด้วย semantic retrieval ตาม baseline ที่กำหนด
2. Text Hybrid RAG ด้วย BM25 + BGE-M3 และ RRF
3. วิธีที่เสนอ: hybrid retrieval เพิ่ม structured facts/rules และ answer verification

รายงาน retrieval แยกจาก answer quality: Recall@k หรือ Hit@k และ MRR; ความถูกต้องคำตอบและตัวเลข; citation/page accuracy; ความถูกต้องของการถามกลับหรืองดตอบ; latency ที่วัดจริง แยก RAGAS เป็นตัวชี้วัดเสริม ไม่ใช้ unit tests แทนผลวิจัย

## เงื่อนไขการตรวจรับ MVP

- คำถามกำกวมหรือไม่มีชื่อทุนชัดเจนต้องถามกลับหรือแจ้งข้อมูลไม่พอ ไม่เดา
- chunk, fact, rule ที่ยัง draft/unverified ต้องไม่เข้าบริบทให้ LLM
- ปีที่ถามต้องตรงกับปีของหลักฐาน และ ceiling ต้องไม่ถูกนำเสนอเป็น allocated amount
- คำตอบ LLM ที่มีตัวเลข/claim/citation ไม่ตรงหลักฐานต้องถูกปิดกั้นทั้งข้อความ
- LLM timeout, Ollama ไม่พร้อม, retrieval vector ล้มเหลว ต้อง fallback แบบควบคุมได้โดยไม่ลดเงื่อนไขการตรวจ
- การเผยแพร่ต้องเกิดหลังผู้ตรวจระบุชื่อ เทียบต้นฉบับ และยืนยันหน้าที่เกี่ยวข้อง
