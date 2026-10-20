"""Readiness check: what is loaded, what still waits for a person, and whether the models answer."""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from app.modules.knowledge import corpus as store
from app.infrastructure.ollama import legacy as llm
from app.modules.documents import review

print(f"  ข้อความ {len(store.CHUNKS)} ชิ้น  ตัวเลข {len(store.FACTS)} แถว  กฎ {len(store.RULES)} ข้อ  นิยาม {len(store.DEFINITIONS)} คำ")
pending = review.pending()
print(f"  รอเจ้าหน้าที่ตรวจ: ตัวเลข {len(pending['facts'])} แถว  ข้อความ {pending['chunk_count']} ชิ้น")
for doc, count in sorted(pending["chunks_by_doc"].items()):
    print(f"      {doc}: {count} ชิ้น")
print(f"  หน้า /admin: {'เปิด' if review.enabled() else 'ปิด ยังไม่ได้ตั้ง RISE_ADMIN_TOKEN'}")
print(f"  semantic embedding: {'Ollama ตอบรับ' if llm.available() else 'ใช้ trigram สำรอง (Ollama ติดต่อไม่ได้)'}")
