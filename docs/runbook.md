# RISE runbook

## Docker

ตั้ง `.env` ตาม `.env.example` โดยใช้ `RISE_ANSWER_MODEL=qwen3:4b-instruct-2507-q4_K_M`, `RISE_EMBED_MODEL=bge-m3`, `RISE_ANSWER_ENABLED=true`, `RISE_SLOT_LLM_FALLBACK=true` และกำหนด `RISE_ADMIN_TOKEN`

```sh
docker compose up --build -d
docker compose --profile setup run --rm models
docker compose exec api python scripts/model_smoke.py
```

หน้าแชต `/`, ผู้ดูแล `/admin`, ฐานข้อมูล `/db`, Swagger `/docs`, สถานะ `/api/status`, readiness `/api/ready` ค่า database URL ต้องกำหนดจึงจะใช้งาน API ได้ ไม่มี runtime ที่ถอยไปใช้ JSON

Compose ทำ migration และ insert-only seed ก่อน API/worker ข้อมูลการอนุมัติเดิมต้องคงอยู่ PDF, PostgreSQL, ChromaDB และ Ollama ใช้ named volumes แยกกัน อย่าใช้ `down -v` เมื่อจะรักษาข้อมูล

```sh
# หลังเปลี่ยนโมเดลหรือ environment
docker compose --profile setup run --rm models
docker compose up -d --build --force-recreate api worker
# งานบำรุงรักษา
docker compose logs --tail=100 api worker
docker compose run --rm migrate
docker compose run --rm seed
docker compose exec api python scripts/export_openapi.py
```

setup ติดตั้งและ warm โมเดลก่อนทดสอบ query โดยให้เวลาโหลด cold สูงสุด 180 วินาทีต่อโมเดล แยกจากเวลาตอบคำถาม 35 วินาที Ollama เปิดให้เก็บสองโมเดลใน RAM พร้อมกันและประมวลผลทีละคำขอ เพื่อลดการโหลด BGE-M3/Qwen สลับกัน หากปล่อยว่างเกิน 30 นาทีอาจต้อง warm ใหม่

API, worker และ setup ใช้ชื่อโมเดลเดียวกัน Ollama ภายใน Docker อยู่ที่ `http://ollama:11434` การ restart อย่างเดียวไม่รับ environment ใหม่ `/api/status` แสดงค่าที่ runtime ใช้อยู่ ส่วน `processing` ของคำตอบบอกว่าเรียกโมเดลจริงหรือใช้ fallback พร้อมเหตุผล ไม่มีการบันทึก credentials ลง response

## ตรวจและเผยแพร่เอกสาร

อัปโหลด PDF ใน `/admin` รอ job อ่านข้อความ จากนั้นเปิด PDF เทียบกับทุกหน้า แก้ข้อความ ระบุผู้ตรวจ และยืนยันครบทุกหน้า ก่อนกดเผยแพร่ เอกสารที่ยังอ่านไม่ครบหรือมี OCR error เผยแพร่ไม่ได้ การแก้ข้อความและเผยแพร่ไม่สร้าง facts/rules อัตโนมัติ

สร้าง facts/rules ผ่านฟอร์มพร้อมระบุ quote และ version/page ที่รองรับ ตรวจและอนุมัติแต่ละรายการแยกกัน การเผยแพร่เพิ่ม corpus revision และเข้าคิว vector index แสดงสถานะ jobs แยกจากเอกสาร BM25 ใช้ได้ระหว่างรอ index ห้าม retry จนเขียนทับงานที่คนแก้แล้ว ให้สร้าง version ใหม่เมื่อต้องอ่านต้นฉบับซ้ำ

## ทดสอบ

ใช้ฐานแยกชื่อ `rise_test` เท่านั้น PostgreSQL suite ล้างข้อมูลในฐานทดสอบ ไม่ใช้ฐาน `rise` ของแอป:

```sh
RISE_TEST_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/rise_test \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_postgres.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_unified_query.py tests/test_runtime_context.py tests/test_mvp_domain.py -q
python scripts/export_openapi.py
```

การทดสอบครอบคลุม approval gates, version/citation, model fallback, context, candidate rejection และ revision fencing การทดลองจริงกับ Ollama ต้องแยก latency ตอน cold/warm และ fallback จากคะแนนคุณภาพ ห้ามถือ model smoke test หรือ regression suite เป็นผลวิจัย

## การย้ายจาก API เดิม

เส้นทาง `/v1/*`, `/v2/*`, `/api/update`, `/healthz` และ SSE เลิกให้บริการ ใช้ `/api/*` ตาม Swagger เท่านั้น ใช้ `/api/documents` นำเข้า PDF และ `/api/status` แทน health endpoint เดิม feedback ใช้ `request_id` แทนการส่งคำถามและคำตอบซ้ำ

ไม่มี migration สำหรับการเปลี่ยน prefix นี้ ไม่ลบ rows หรือ volumes หาก rollback ให้กลับ image/โค้ดก่อนหน้าและ recreate API/worker โดยรักษา volumes เดิม ประวัติ query ใหม่มี fields เพิ่มใน JSON ซึ่งโค้ดเดิมสามารถเก็บไว้ได้

## โมเดล Thinking และสิทธิ์ ChromaDB

ใช้ tag `qwen3:4b-instruct-2507-q4_K_M` อย่างชัดเจน รุ่น Thinking-2507 ไม่เหมาะกับการสั่งปิด thinking เพื่อให้ตอบในงบเวลาสั้น ดู [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) และ [Ollama tag](https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M) สคริปต์ smoke คืน JSON แยกผล embedding/answer และ exit code 1 เมื่อขั้นใดล้มเหลว แทนการทิ้งรายงานทั้งหมดเป็น traceback

หาก volume เดิมของ ChromaDB สร้างด้วย root แล้วพบ `unable to open database file` ให้ซ่อมเจ้าของไฟล์ให้ตรงกับ UID 10001 ของ API/worker โดยไม่ลบ volume:

```sh
docker compose run --rm --no-deps --user 0 --entrypoint sh api \
  -c 'mkdir -p /srv/rise/data/chroma_db && chown -R 10001:10001 /srv/rise/data/chroma_db'
docker compose up -d --build api worker
```

Dockerfile เตรียม directory นี้ด้วยเจ้าของที่ถูกต้องสำหรับ volume ใหม่แล้ว หากดัชนีเปิดไม่ได้ API ต้องใช้ BM25 fallback แทน HTTP 500
