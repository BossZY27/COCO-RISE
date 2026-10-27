"""Connection/latency smoke check only; never a claim of answer quality."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.settings import Settings
from app.infrastructure.ollama.client import OllamaClient

if __name__ == "__main__":
    client = OllamaClient(Settings())
    report = {}
    try:
        start = time.perf_counter()
        vector = client.embed(["ทุนวิจัย"], timeout=120)[0]
        report["embedding"] = {"dimensions": len(vector), "latency_ms": round((time.perf_counter()-start)*1000, 1)}
        start = time.perf_counter()
        answer, revision = client.candidate("เพดานทุนวิจัยสถาบัน ไม่เกิน 25,000 บาท", [])
        report["answer"] = {"completed": bool(answer), "model_revision": revision,
                            "latency_ms": round((time.perf_counter()-start)*1000, 1)}
        print(json.dumps(report, ensure_ascii=False))
    finally:
        client.close()
