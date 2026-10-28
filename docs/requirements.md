# ข้อกำหนดตามข้อเสนอ

| ความสามารถ | เส้นทางในระบบ |
| --- | --- |
| เตรียม PDF สแกนด้วย OCR และตรวจต้นฉบับ | documents/pages, worker, review/publish |
| ถามตอบภาษาไทยพร้อมอ้างอิง | `/api/query`, context, versioned citations |
| Hybrid retrieval และข้อมูลเชิงโครงสร้าง | BM25 + BGE-M3/ChromaDB + RRF และ PostgreSQL facts/rules |
| LLM เรียบเรียงจากหลักฐาน | Ollama candidate หลังเลือกข้อมูล พร้อม verifier และ template fallback |
| ตรวจข้อมูลก่อนตอบ | verified/published gates, candidate checks, corpus revision fencing |
| ประเมินผลสามวิธี | แยก retrieval, answer, citation, abstention และ latency ตาม evaluation protocol |

HTML และการเชื่อมฐานข้อมูลบุคลากรเป็นระยะถัดไป ไม่แนะนำทุนเฉพาะบุคคล ไม่ตรวจร่างข้อเสนอ และไม่รับรองสิทธิ์หรือการได้รับทุน
