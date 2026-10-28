"""Input limits and a per-user rate limit.

On this hardware one answer occupies the model for ten to twenty seconds. One person holding the enter
key would therefore lock everyone else out, so each user gets a small budget per minute. The window is
kept in memory, which is correct for one process; behind two workers it becomes per-worker, and that is
the point where it belongs in Redis.
"""
import os
import re
import threading
import time
from collections import deque

MAX_QUESTION_CHARS = int(os.getenv("RISE_MAX_QUESTION", "400"))
MAX_HISTORY_TURNS = int(os.getenv("RISE_MAX_HISTORY", "8"))
MAX_HISTORY_ITEM_CHARS = int(os.getenv("RISE_MAX_HISTORY_ITEM", str(MAX_QUESTION_CHARS)))
MAX_REQUEST_BODY_BYTES = int(os.getenv("RISE_MAX_REQUEST_BODY_BYTES", "65536"))
RATE_LIMIT = int(os.getenv("RISE_RATE_LIMIT", "12"))          # questions per window
RATE_WINDOW = float(os.getenv("RISE_RATE_WINDOW", "60"))      # seconds
CLIENT_RATE_LIMIT = int(os.getenv("RISE_CLIENT_RATE_LIMIT", str(RATE_LIMIT)))
CLIENT_RATE_WINDOW = float(os.getenv("RISE_CLIENT_RATE_WINDOW", str(RATE_WINDOW)))
MAX_FEEDBACK_QUESTION_CHARS = int(
    os.getenv("RISE_MAX_FEEDBACK_QUESTION", str(MAX_QUESTION_CHARS))
)
MAX_FEEDBACK_ANSWER_CHARS = int(os.getenv("RISE_MAX_FEEDBACK_ANSWER", "8000"))
MAX_FEEDBACK_NOTE_CHARS = int(os.getenv("RISE_MAX_FEEDBACK_NOTE", "400"))
FEEDBACK_RATE_LIMIT = int(os.getenv("RISE_FEEDBACK_RATE_LIMIT", "8"))
FEEDBACK_RATE_WINDOW = float(os.getenv("RISE_FEEDBACK_RATE_WINDOW", "60"))
FEEDBACK_CLIENT_RATE_LIMIT = int(
    os.getenv("RISE_FEEDBACK_CLIENT_RATE_LIMIT", str(FEEDBACK_RATE_LIMIT))
)
FEEDBACK_CLIENT_RATE_WINDOW = float(
    os.getenv("RISE_FEEDBACK_CLIENT_RATE_WINDOW", str(FEEDBACK_RATE_WINDOW))
)
MAX_RATE_BUCKETS = max(1, int(os.getenv("RISE_RATE_BUCKET_CAP", "2048")))
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_seen: dict[str, deque] = {}
_client_seen: dict[str, deque] = {}
_feedback_seen: dict[str, deque] = {}
_feedback_client_seen: dict[str, deque] = {}
_rate_lock = threading.Lock()


class RequestBodyLimitMiddleware:
    """Reject oversized HTTP request bodies before FastAPI/Pydantic buffers JSON.

    PDF endpoints receive their own larger caps from main.py; ordinary public and
    admin JSON stays small. The body is replayed to the downstream ASGI app only
    after the bounded read completes.
    """

    def __init__(self, app, default_max: int = MAX_REQUEST_BODY_BYTES,
                 path_limits: dict[str, int] | None = None):
        self.app = app
        self.default_max = default_max
        self.path_limits = path_limits or {}

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        pdf_upload = path == "/api/documents" or (path.startswith("/api/documents/") and path.endswith("/versions"))
        maximum = self.path_limits.get(path, 15 * 1024 * 1024 if pdf_upload else self.default_max)
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length:
            try:
                if int(raw_length) > maximum:
                    await self._reject(send, maximum)
                    return
            except ValueError:
                await self._reject(send, maximum)
                return

        messages, total = [], 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") == "http.disconnect":
                break
            total += len(message.get("body", b""))
            if total > maximum:
                await self._reject(send, maximum)
                return
            if not message.get("more_body", False):
                break

        async def replay():
            if messages:
                return messages.pop(0)
            return await receive()

        await self.app(scope, replay, send)

    @staticmethod
    async def _reject(send, maximum: int) -> None:
        import json

        body = json.dumps(
            {"detail": f"คำขอมีขนาดใหญ่เกิน {maximum} ไบต์"},
            ensure_ascii=False,
        ).encode("utf-8")
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json; charset=utf-8"),
                                (b"content-length", str(len(body)).encode("ascii")),
                                (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})


def clean_question(raw: str) -> tuple[str, str | None]:
    """Return the question to use, plus a reason to refuse it."""
    text = CONTROL_CHARS.sub(" ", raw or "")
    text = " ".join(text.split())
    if not text:
        return "", "กรุณาพิมพ์คำถาม"
    if len(text) > MAX_QUESTION_CHARS:
        return "", f"คำถามยาวเกิน {MAX_QUESTION_CHARS} ตัวอักษร กรุณาถามให้สั้นลง"
    return text, None


def clean_history(raw: list[str] | None) -> list[str]:
    items = [
        " ".join(CONTROL_CHARS.sub(" ", h).split())[:MAX_HISTORY_ITEM_CHARS]
        for h in (raw or [])
    ]
    return [h for h in items if h][-MAX_HISTORY_TURNS:]


THAI_NATIONAL_ID_RE = re.compile(r"(?<!\d)\d(?:[- ]?\d){12}(?!\d)")
THAI_PHONE_RE = re.compile(
    r"(?<!\d)(?:0[689]\d[- ]?\d{3}[- ]?\d{4}|0[23457][- ]?\d{3}[- ]?\d{4})(?!\d)"
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SELF_DECLARED_NAME_RE = re.compile(
    r"(?:ผมชื่อ|ฉันชื่อ|ดิฉันชื่อ|หนูชื่อ|ชื่อ-นามสกุล)\s*(?:นาย|นาง|นางสาว|น\.ส\.|ดร\.|ผศ\.|รศ\.|ศ\.|อ\.)?\s*[ก-๙A-Za-z]+(?:\s+[ก-๙A-Za-z]+)?"
)


def redact_pii(raw: str | None) -> str:
    """Mask personal identifiers (Thai ID, phone, email, self-declared full name) before LLM calls or audit logs."""
    if not raw:
        return ""
    text = EMAIL_RE.sub("[ปกปิดอีเมล]", raw)
    text = THAI_NATIONAL_ID_RE.sub("[ปกปิดเลขประจำตัว]", text)
    text = THAI_PHONE_RE.sub("[ปกปิดเบอร์โทรศัพท์]", text)
    text = SELF_DECLARED_NAME_RE.sub("[ปกปิดชื่อส่วนบุคคล]", text)
    return text


def contains_pii(raw: str | None) -> bool:
    """Return True when the text contains recognizable personal identifiers."""
    if not raw:
        return False
    return bool(
        EMAIL_RE.search(raw)
        or THAI_NATIONAL_ID_RE.search(raw)
        or THAI_PHONE_RE.search(raw)
        or SELF_DECLARED_NAME_RE.search(raw)
    )


def clean_feedback_text(raw: str) -> str:
    """Remove log-hostile control characters without changing the user's wording."""
    return redact_pii(CONTROL_CHARS.sub(" ", raw or "").strip())


def _take(
    buckets: dict[str, deque], user_id: str, budget: int, window_seconds: float
) -> tuple[bool, float]:
    now = time.monotonic()
    with _rate_lock:
        # Delete expired identities before inserting a new one. This keeps a caller from
        # exhausting process memory by sending an unlimited stream of made-up IDs.
        for key, events in list(buckets.items()):
            while events and now - events[0] > window_seconds:
                events.popleft()
            if not events:
                del buckets[key]

        key = user_id or "anonymous"
        if key not in buckets and len(buckets) >= MAX_RATE_BUCKETS:
            oldest = min(buckets, key=lambda item: buckets[item][-1])
            del buckets[oldest]

        if budget <= 0:
            return False, max(0.0, window_seconds)

        window = buckets.setdefault(key, deque())
        while window and now - window[0] > window_seconds:
            window.popleft()
        if len(window) >= budget:
            return False, round(max(0.0, window_seconds - (now - window[0])), 1)
        window.append(now)
    return True, 0.0


def take(user_id: str) -> tuple[bool, float]:
    """Spend one query slot and report when the next slot becomes available."""
    return _take(_seen, user_id, RATE_LIMIT, RATE_WINDOW)


def take_client(client_host: str) -> tuple[bool, float]:
    """Spend an IP-bound query slot that cannot be reset with a forged persona/header."""
    return _take(_client_seen, client_host, CLIENT_RATE_LIMIT, CLIENT_RATE_WINDOW)


def take_feedback(user_id: str) -> tuple[bool, float]:
    """Spend one feedback slot without consuming the user's question budget."""
    return _take(
        _feedback_seen,
        user_id,
        FEEDBACK_RATE_LIMIT,
        FEEDBACK_RATE_WINDOW,
    )


def take_feedback_client(client_host: str) -> tuple[bool, float]:
    """Spend an IP-bound feedback slot in addition to the logical-user slot."""
    return _take(
        _feedback_client_seen,
        client_host,
        FEEDBACK_CLIENT_RATE_LIMIT,
        FEEDBACK_CLIENT_RATE_WINDOW,
    )
