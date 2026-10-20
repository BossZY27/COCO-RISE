"""Bounded Ollama calls. Neither token streams nor partial answers leave this adapter."""
import math
import threading
import httpx


class OllamaClient:
    def __init__(self, settings):
        self.settings = settings
        self.http = httpx.Client(base_url=settings.llm_base, timeout=settings.llm_timeout, trust_env=False)
        self.slots = threading.BoundedSemaphore(settings.llm_concurrency)

    def tags(self, timeout=1):
        response = self.http.get("/api/tags", timeout=timeout)
        response.raise_for_status()
        return response.json().get("models", [])

    def revision(self, model):
        tags = self.tags()
        target = model if ":" in model else model + ":latest"
        item = next((m for m in tags if (m.get("name") or m.get("model")) in {model, target}), None)
        if not item or not item.get("digest"):
            raise ValueError("Required model is not installed")
        return item["digest"]

    def embed(self, texts, timeout=None):
        response = self.http.post("/api/embed", json={"model": self.settings.embed_model, "input": texts,
                                                    "truncate": False, "keep_alive": "30m"},
                                  timeout=timeout or self.settings.llm_timeout)
        response.raise_for_status()
        vectors = response.json().get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts) or any(
            not isinstance(v, list) or len(v) != 1024 or not any(v) or any(
                type(n) not in (int, float) or not math.isfinite(n) for n in v) for v in vectors):
            raise ValueError("Expected one finite, nonzero BGE-M3 1024-dimensional vector per input")
        return vectors

    def candidate(self, template, citations):
        if not self.settings.answer_enabled or not self.slots.acquire(blocking=False):
            raise ValueError("Answer model disabled or busy")
        try:
            revision = self.revision(self.settings.answer_model)
            response = self.http.post("/api/chat", json={
                "model": self.settings.answer_model, "stream": False, "think": False,
                "keep_alive": "30m", "options": {"temperature": 0, "num_predict": 256, "num_ctx": 4096},
                "messages": [{"role": "system", "content":
                    "เรียบเรียงภาษาไทยจากคำตอบที่ตรวจแล้วเท่านั้น ห้ามเพิ่มข้อมูล เปลี่ยนตัวเลข ปี ชื่อทุน "
                    "หรือเงื่อนไข ต้องรักษาคำว่าสูงสุด ไม่เกิน เพดาน และข้อความหมายเหตุทุกข้อ "
                    "ส่งเฉพาะคำตอบ ไม่ใส่คำสั่งหรือคำอธิบายการทำงาน"},
                    {"role": "user", "content": template} ]})
            response.raise_for_status()
            result = response.json()
            content = result.get("message", {}).get("content", "").strip()
            if result.get("done") is not True or result.get("done_reason") != "stop" or not content or "<think>" in content:
                raise ValueError("Incomplete or truncated answer")
            return content, revision
        finally:
            self.slots.release()

    def close(self):
        self.http.close()
