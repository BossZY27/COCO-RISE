"""Client for the local Ollama runtime.

Measured on this machine (CPU only, no GPU offload):
  thai-qwen:latest   about 6 tokens per second, so an answer of 70 tokens takes roughly 12 seconds
  qwen3:4b           about 9 tokens per second, but it writes a thinking preamble, so it is used for JSON only
  bge-m3:latest      about 1.1 seconds for one query embedding once the model is warm

Two consequences shape the whole design: keep the models warm, and keep the answer short and streamed.
"""
import json
import os
import re
import threading
import time

import httpx

BASE = os.getenv("RISE_LLM_BASE", "http://127.0.0.1:11434")
MAIN = os.getenv("RISE_ANSWER_MODEL", "qwen3:1.7b")
SMALL = os.getenv("RISE_LLM_SMALL", "qwen3:4b")
EMBED_MODEL = os.getenv("RISE_EMBED_MODEL", "bge-m3:latest")
KEEP_ALIVE = os.getenv("RISE_KEEP_ALIVE", "30m")
MAX_TOKENS = int(os.getenv("RISE_MAX_TOKENS", "110"))
TIMEOUT = float(os.getenv("RISE_LLM_TIMEOUT", "40"))
MODE = os.getenv("RISE_LLM_MODE", "local")          # local uses Ollama, template skips it
# Readers may pick MAIN or a model named here. Being installed is not enough: a 27B model on this CPU
# outlasts TIMEOUT and holds the single model slot while it does, so the operator opts each one in.
ANSWER_MODELS = tuple(name.strip() for name in os.getenv("RISE_ANSWER_MODELS", "").split(",") if name.strip())
# Ollama unloads a model after KEEP_ALIVE of silence, and reloading the answer model costs about
# fifteen seconds. A quiet office therefore makes the next person wait. A tiny ping keeps it in.
KEEP_WARM_MINUTES = float(os.getenv("RISE_KEEP_WARM_MINUTES", "10"))
THINK_TAG = re.compile(r"<think>.*?</think>", re.S)
# The asyncio semaphore lives in pipeline.py, but a disconnected streaming client
# can release it while its blocking worker thread is still reading Ollama.  This
# process-wide lock keeps the actual chat requests serialized until HTTP really ends.
CHAT_LOCK = threading.Lock()


def enabled() -> bool:
    return MODE == "local"


def available(timeout: float = 1.5) -> bool:
    if not enabled():
        return False
    try:
        httpx.get(f"{BASE}/api/version", timeout=timeout)
        return True
    except Exception:
        return False


def installed_chat_models() -> list[dict]:
    """List local answer models without loading each one to inspect it."""
    response = httpx.get(f"{BASE}/api/tags", timeout=5)
    response.raise_for_status()
    models = []
    for item in response.json().get("models", []):
        name = item.get("name") or item.get("model")
        family = str(item.get("details", {}).get("family", "")).lower()
        if (not name or name == EMBED_MODEL or "embed" in name.lower()
                or family in {"bert", "nomic-bert"}):
            continue
        models.append({"name": name, "size": item.get("size", 0)})
    return sorted(models, key=lambda item: item["name"])


def allowed_model_names() -> set[str]:
    """The operator's list: the answer model plus any named in RISE_ANSWER_MODELS."""
    return {MAIN, *ANSWER_MODELS}


def answer_models() -> list[dict]:
    """The models on the operator's list that are installed, for the page to offer."""
    allowed = allowed_model_names()
    return [item for item in installed_chat_models() if item["name"] in allowed]


def _body(model: str, messages: list[dict], max_tokens: int, json_mode: bool) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "keep_alive": KEEP_ALIVE, "think": False,
            "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 2048, "top_p": 0.9}}
    if json_mode:
        body["format"] = "json"
    return body


def chat(messages: list[dict], model: str = "", max_tokens: int = 0, json_mode: bool = False,
         timeout: float = 0.0) -> str:
    """One shot answer. Raises on transport errors so the caller can fall back to the template."""
    body = _body(model or MAIN, messages, max_tokens or MAX_TOKENS, json_mode)
    with CHAT_LOCK:
        data = httpx.post(f"{BASE}/api/chat", json=body, timeout=timeout or TIMEOUT).json()
    return THINK_TAG.sub("", data.get("message", {}).get("content", "")).strip()


def stream_chat(messages: list[dict], model: str = "", max_tokens: int = 0, timeout: float = 0.0):
    """Yield the answer piece by piece so the reader sees text within a few seconds."""
    body = _body(model or MAIN, messages, max_tokens or MAX_TOKENS, False)
    body["stream"] = True
    with CHAT_LOCK:
        with httpx.stream("POST", f"{BASE}/api/chat", json=body, timeout=timeout or TIMEOUT) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    return


def embed(texts: list[str], timeout: float = 30.0) -> list[list[float]]:
    data = httpx.post(f"{BASE}/api/embed",
                      json={"model": EMBED_MODEL, "input": texts, "keep_alive": KEEP_ALIVE},
                      timeout=timeout).json()
    return data["embeddings"]


def warmup() -> dict:
    """Load the models into memory before the first question, otherwise it pays the load time."""
    report = {}
    if not enabled():
        return {"mode": "template"}
    # embed first, chat last. If the machine runs short of memory Ollama evicts the model it loaded
    # earliest, and the answer model is the one worth keeping: reloading it costs about fifteen
    # seconds, while the embedding model reloads in one.
    for label, call in (("embed", lambda: embed(["พร้อม"])),
                        ("chat", lambda: chat([{"role": "user", "content": "ตอบว่า พร้อม"}], max_tokens=4))):
        t0 = time.perf_counter()
        try:
            call()
            report[label] = round(time.perf_counter() - t0, 1)
        except Exception as exc:
            report[label] = f"ไม่สำเร็จ: {type(exc).__name__}"
    return report
