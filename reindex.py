"""Embed every chunk once so the vector path is ready before the first question."""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from app.modules.knowledge import corpus as store
from app.infrastructure.ollama import legacy as llm

t0 = time.perf_counter()
print("ollama ตอบ:", llm.available())
report = store.build_embeddings(
    progress=lambda done, total: print(f"  ฝังแล้ว {done}/{total} ชิ้น", flush=True))
print(f"chunks {len(store.CHUNKS)}  facts {len(store.FACTS)}  funds {len(store.FUNDS)}")
print(f"embeddings {report}  ใช้เวลา {time.perf_counter() - t0:.1f} วิ")
print("vectors_ready:", store.vectors_ready())
