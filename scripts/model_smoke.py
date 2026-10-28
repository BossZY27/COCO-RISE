"""Connection/latency smoke check only; never a claim of answer quality."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.settings import Settings
from app.infrastructure.ollama.client import OllamaClient


def run_smoke(client):
    report = {}
    for stage in ("embedding", "answer"):
        started = time.perf_counter()
        result = {"completed": False, "model": client.settings.embed_model if stage == "embedding" else client.settings.answer_model}
        try:
            if stage == "embedding":
                result["dimensions"] = len(client.embed(["ทุนวิจัย"], timeout=120)[0])
            else:
                answer, revision = client.candidate("เพดานทุนวิจัยสถาบัน ไม่เกิน 25,000 บาท", [])
                result["model_revision"] = revision
                if not answer:
                    raise ValueError("Empty answer")
            result["completed"] = True
        except Exception as exc:
            result["error"] = type(exc).__name__
            result["hint"] = "Check model variant, Ollama logs and warmup; API query uses verified fallback on model failure"
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        report[stage] = result
    report["ok"] = all(report[stage]["completed"] for stage in ("embedding", "answer"))
    return report


def main():
    client = OllamaClient(Settings())
    try:
        report = run_smoke(client)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["ok"] else 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
