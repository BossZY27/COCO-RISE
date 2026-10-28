# COCO-RISE

ระบบถามตอบข้อมูลทุนวิจัยภาษาไทยจาก PDF ที่ผู้ดูแลตรวจแล้ว ใช้ FastAPI, PostgreSQL, ChromaDB และ Ollama โดยมี API ชุดเดียวที่ `/api/*` ยังไม่เชื่อมฐานข้อมูลบุคลากร ไม่ตัดสินสิทธิ์รายบุคคล และไม่รับรองผลการได้รับทุน

## เริ่มด้วย Docker

คัดลอก `.env.example` เป็น `.env` หากยังไม่มี แล้วตั้ง `RISE_ADMIN_TOKEN` และค่าโมเดล:

```dotenv
RISE_ANSWER_MODEL=qwen3:4b-instruct-2507-q4_K_M
RISE_EMBED_MODEL=bge-m3
RISE_ANSWER_ENABLED=true
RISE_SLOT_LLM_FALLBACK=true
```

```sh
docker compose up --build -d
docker compose --profile setup run --rm models
```

เปิด [แชต](http://localhost:8077/), [ตรวจทานข้อมูล](http://localhost:8077/admin), [ฐานข้อมูล](http://localhost:8077/db) หรือ [Swagger](http://localhost:8077/docs) สถานะระบบอยู่ที่ `/api/status` และ readiness อยู่ที่ `/api/ready`

เมื่อเปลี่ยน `.env` ให้ recreate API และ worker เพื่อรับค่าใหม่:

```sh
docker compose up -d --build --force-recreate api worker
```

Compose ตั้ง Ollama เป็น `http://ollama:11434` ภายใน network และใช้ชื่อโมเดลเดียวกันใน API, worker และขั้นติดตั้งโมเดล ค่า `RISE_LLM_SMALL` และ `RISE_ANSWER_MODE` ไม่ควบคุม API ปัจจุบัน

## การทำงาน

1. วิเคราะห์ชื่อทุน ชั้น ปีงบ และเจตนาด้วยกฎ ใช้ LLM ช่วยเมื่อยังระบุไม่ได้ บริบทสนทนาใช้แก้คำอ้างอิงเท่านั้น
2. ค้นข้อความด้วย BM25 และ BGE-M3 ผ่าน ChromaDB รวมอันดับด้วย RRF พร้อมค้น facts/rules ตามทุน ปี และประเภทวงเงิน
3. โปรแกรมเลือกข้อมูลที่ตรวจแล้วและประกอบ template; LLM เรียบเรียงคำตอบที่มีหลักฐานเพียงพอ
4. ตรวจตัวเลข ชื่อทุน ปี ประเภทวงเงิน หน่วย เงื่อนไข และอ้างอิง หาก candidate ไม่ผ่านใช้ template ที่ตรวจแล้ว หากหลักฐานไม่ผ่านหรือเปลี่ยนรุ่นระหว่างตอบจะบล็อก

การตรวจภาษาไทยเป็นแบบอนุรักษ์นิยม: ข้อความที่มีตัวเลขและเงื่อนไขสำคัญต้องคงเดิม จึงอาจใช้ template สำรองบ่อย การผ่านกฎไม่พิสูจน์ว่าตรวจความหมายได้ครบทุกกรณี

เมื่อโมเดลหรือ vector index ไม่พร้อม ระบบใช้ BM25 และ template พร้อมแจ้งเหตุผลใน `processing` ไม่แทน BGE-M3 ด้วย trigram แล้วอ้างว่าใช้เวกเตอร์สำเร็จ งบงานโมเดลในคำถามหนึ่งรวม 35 วินาที หน้าเว็บรอได้ 40 วินาที

## เอกสารและข้อมูล

อัปโหลด PDF → worker อ่านข้อความ/OCR → ผู้ดูแลแก้และเทียบต้นฉบับครบทุกหน้า → ยืนยัน → เผยแพร่ → เข้าคิวสร้างดัชนีเวกเตอร์ ระหว่างรอ index ใช้ BM25 ได้ การเผยแพร่ PDF ไม่อนุมัติ facts/rules อัตโนมัติ ต้องสร้างรายการพร้อม quote และ version/page แล้วตรวจและอนุมัติแยก

PostgreSQL เก็บเอกสาร รุ่น ข้อความ facts/rules jobs และประวัติ ChromaDB เก็บเวกเตอร์ ส่วน PDF อยู่ใน private volume ข้อมูล JSON ใช้ seed และการทดลอง offline ไม่ใช่ fallback runtime ข้อมูลที่ยังไม่อนุมัติใช้ตอบไม่ได้ เพดานกลางและวงเงินจัดสรรรายปีต้องแยกกันเสมอ

การนำเข้า HTML และการเชื่อมข้อมูลบุคลากรเป็นงานระยะถัดไป

## API

ใช้ `POST /api/query` พร้อม `question`, `history` และ `context` แบบเลือกส่ง ผลตอบกลับมี `request_id`, `status`, `answer`, `answer_source`, `citations`, `verification`, `processing`, `context` และ `timings_ms`

```sh
curl http://localhost:8077/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"เพดานทุนวิจัยในชั้นเรียนเท่าไร"}'
```

ส่ง feedback ด้วย `request_id` ไป `/api/feedback` งานผู้ดูแลต้องส่ง `X-Admin-Token` รวมถึง `/api/documents`, `/api/versions`, `/api/facts`, `/api/rules`, `/api/jobs`, `/api/recent` และ `/api/db/overview` เปิดแหล่งอ้างอิงที่เผยแพร่แล้วผ่าน `/api/sources/{version_id}/pages/{page}`

เส้นทาง `/v1/*`, `/v2/*`, `/api/update`, `/healthz` และ SSE เดิมเลิกใช้แล้ว ต้องย้ายผู้เรียกตาม schema ใน Swagger ประวัติแชตเก่าเปิดอ่านได้ แต่รายการที่ไม่มี `request_id` ส่ง feedback แบบใหม่ไม่ได้

ดู [สถาปัตยกรรม](docs/architecture.md), [คู่มือรันและทดสอบ](docs/runbook.md) และ [แนวทางประเมินผล](docs/evaluation.md) ผลทดสอบโค้ดและ model smoke test ไม่ใช่คะแนนวิจัย
