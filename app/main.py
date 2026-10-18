"""FastAPI entry point: chat page, blocking query, streaming query, health check.

On startup the app prepares semantic-search vectors. Fact-bearing answers stay on
the verified code renderer; no free-form answer model is loaded in safe mode.
"""
import ipaddress
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import audit, identity, ingestion, limits, llm, pdf_inspection, pipeline, store
from .api_admin import router as admin_router
from .schemas import Feedback, Query


def _enabled(name: str, default: str = "off") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "on", "true", "yes"}


def _trusted_hosts() -> list[str]:
    raw = os.getenv("IRFA_TRUSTED_HOSTS", "127.0.0.1,localhost,testserver")
    hosts = [host.strip() for host in raw.split(",") if host.strip()]
    if not hosts:
        raise RuntimeError("IRFA_TRUSTED_HOSTS ต้องมี host อย่างน้อยหนึ่งค่า")
    if any("*" in host or "://" in host or "/" in host for host in hosts):
        raise RuntimeError("IRFA_TRUSTED_HOSTS ต้องเป็นชื่อ host หรือ IP แบบเจาะจงและห้ามใช้ *")
    return hosts


def _trusted_proxy_ips() -> frozenset[str]:
    """Return exact proxy IPs allowed to assert an authenticated user header."""
    raw = os.getenv("IRFA_TRUSTED_PROXY_IPS", "127.0.0.1,::1")
    addresses: set[str] = set()
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            addresses.add(str(ipaddress.ip_address(value)))
        except ValueError as exc:
            raise RuntimeError("IRFA_TRUSTED_PROXY_IPS ต้องระบุเฉพาะ IP แบบเจาะจง") from exc
    if not addresses:
        raise RuntimeError("IRFA_TRUSTED_PROXY_IPS ต้องมี IP อย่างน้อยหนึ่งค่า")
    return frozenset(addresses)


def _proxy_user_id(header_user: str | None, client_host: str) -> str | None:
    """Trust identity only when both the feature and the exact proxy IP allow it."""
    if not TRUST_PROXY_USER or not header_user:
        return None
    try:
        remote = str(ipaddress.ip_address(client_host))
    except ValueError:
        return None
    return header_user if remote in _trusted_proxy_ips() else None


DOCS_ENABLED = _enabled("IRFA_EXPOSE_DOCS", "on")
TRUST_PROXY_USER = _enabled("IRFA_TRUST_PROXY_USER", "off")
STATUS: dict = {"llm": "ปิดการเขียนคำตอบอิสระ", "embeddings": "ยังไม่เริ่ม"}
LOGGER = logging.getLogger(__name__)
_EMBED_WARMUP_LOCK = threading.Lock()
_EMBED_WARMUP_THREAD: threading.Thread | None = None


def _warm_embeddings() -> None:
    """Prepare optional vectors without holding up the public API."""
    try:
        STATUS["embeddings"] = store.build_embeddings()
    except Exception as exc:
        STATUS["embeddings"] = f"ไม่สำเร็จ: {type(exc).__name__} (จะใช้การค้นแบบสำรองแทน)"
        LOGGER.exception("Embedding warmup failed; trigram retrieval remains available")


def _start_embedding_warmup() -> None:
    """Start at most one daemon warmup; retrieval already has a lexical fallback."""
    global _EMBED_WARMUP_THREAD
    with _EMBED_WARMUP_LOCK:
        if _EMBED_WARMUP_THREAD is not None and _EMBED_WARMUP_THREAD.is_alive():
            STATUS["embeddings"] = "กำลังเตรียมในเบื้องหลัง"
            return
        STATUS["embeddings"] = "กำลังเตรียมในเบื้องหลัง"
        _EMBED_WARMUP_THREAD = threading.Thread(
            target=_warm_embeddings,
            name="irfa-embedding-warmup",
            daemon=True,
        )
        _EMBED_WARMUP_THREAD.start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Recover ingestion state and prepare retrieval before accepting traffic."""
    if TRUST_PROXY_USER:
        _trusted_proxy_ips()
    try:
        recovered_jobs = ingestion.resume_incomplete_jobs()
        STATUS["ingestion"] = (
            f"กำลังทำงานต่อ {len(recovered_jobs)} งาน" if recovered_jobs else "ไม่มีงานค้าง"
        )
    except ingestion.IngestionStorageError as exc:
        STATUS["ingestion"] = f"อ่านสถานะงานไม่ได้: {exc}"
        LOGGER.exception("Ingestion restart recovery skipped because persisted state is unreadable")
    _start_embedding_warmup()
    STATUS["llm"] = "ปิดการเขียนคำตอบอิสระ ใช้ข้อความจากข้อมูลที่ตรวจแล้ว"
    yield


app = FastAPI(
    title="COCO-RISE demo",
    version="0.4.0",
    docs_url="/docs" if DOCS_ENABLED else None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
    lifespan=lifespan,
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=_trusted_hosts(),
    www_redirect=False,
)
app.add_middleware(
    limits.RequestBodyLimitMiddleware,
    default_max=limits.MAX_REQUEST_BODY_BYTES,
    path_limits={
        "/v1/admin/inspect-pdf": pdf_inspection.MAX_BYTES,
        "/v1/admin/documents": ingestion.MAX_BYTES,
    },
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Set browser protections that are safe for the local and LAN demo pages."""
    response = await call_next(request)
    headers = response.headers
    headers.setdefault("X-Content-Type-Options", "nosniff")
    headers.setdefault("X-Frame-Options", "DENY")
    headers.setdefault("Referrer-Policy", "no-referrer")
    headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; base-uri 'none'; object-src 'none'; "
        "frame-ancestors 'none'; form-action 'self'; connect-src 'self'; "
        "img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'",
    )
    if (request.url.path == "/v1/recent" or request.url.path == "/admin"
            or request.url.path.startswith("/v1/admin/")):
        headers["Cache-Control"] = "no-store, private"
    return response


@app.exception_handler(ingestion.IngestionStorageError)
async def ingestion_storage_error(_request: Request, exc: ingestion.IngestionStorageError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(admin_router)


def _accept(
    body: Query, header_user: str | None, client_host: str
) -> tuple[str, str, list[str]]:
    """Identity, input limits and the rate limit, applied the same way on both query endpoints."""
    trusted_header = _proxy_user_id(header_user, client_host)
    if not trusted_header and body.persona_id not in store.PERSONAS:
        raise HTTPException(status_code=400, detail="ไม่พบโปรไฟล์สาธิตที่เลือก")
    user_id, _ = identity.resolve(trusted_header, body.persona_id)
    question, refusal = limits.clean_question(body.question)
    if refusal:
        raise HTTPException(status_code=400, detail=refusal)
    allowed, wait = limits.take_client(client_host)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"ถามถี่เกินไป กรุณารออีก {wait} วินาที")
    allowed, wait = limits.take(user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail=f"ถามถี่เกินไป กรุณารออีก {wait} วินาที")
    return user_id, question, limits.clean_history(body.history)




@app.get("/healthz")
def healthz():
    unverified = sum(1 for f in store.FACTS if not store.is_verified(f))
    return {"status": "ok", "chunks": len(store.CHUNKS), "facts": len(store.FACTS), "rules": len(store.RULES),
            "facts_awaiting_review": unverified,
            "answer_mode": pipeline.ANSWER_MODE, "main_model": None, "embed_model": llm.EMBED_MODEL,
            "vectors_ready": store.vectors_ready(), "semantic_backend": pipeline.semantic_backend(),
            "warmup": STATUS}


@app.get("/v1/models")
def models():
    return {"enabled": False, "default": None, "models": [],
            "answer_mode": pipeline.ANSWER_MODE,
            "message": "โหมดปลอดภัยใช้คำตอบที่โค้ดประกอบจากข้อมูลที่ตรวจแล้ว"}


def _selected_model(name: str | None) -> None:
    if name:
        raise HTTPException(status_code=400, detail="โหมดปลอดภัยยังไม่เปิดให้โมเดลเขียนคำตอบอิสระ")
    return None


@app.post("/v1/feedback")
def feedback(
    body: Feedback,
    request: Request,
    x_user_id: str | None = Header(default=None, max_length=128),
):
    """A reader saying the answer was wrong is the cheapest test case the project will ever get."""
    remote = request.client.host if request.client else "anonymous"
    user_id = _proxy_user_id(x_user_id, remote) or remote
    allowed, wait = limits.take_feedback_client(remote)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"ส่งความเห็นถี่เกินไป กรุณารออีก {wait} วินาที",
        )
    allowed, wait = limits.take_feedback(user_id)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"ส่งความเห็นถี่เกินไป กรุณารออีก {wait} วินาที",
        )
    if audit.ENABLED:
        audit.record({"question": limits.clean_feedback_text(body.question),
                      "answer": limits.clean_feedback_text(body.answer),
                      "answer_source": "feedback", "citations": [],
                      "ruling": {"verdict": "helpful" if body.helpful else "not_helpful"},
                      "slots": {"note": limits.clean_feedback_text(body.note)},
                      "timings_ms": {}}, user_id)
    return {"ok": True, "recorded": bool(audit.ENABLED)}




@app.get("/v1/personas")
def personas():
    return list(store.PERSONAS.values())


@app.post("/v1/query")
async def query(
    body: Query, request: Request,
    x_user_id: str | None = Header(default=None, max_length=128),
):
    # The proxy sets X-User-Id after it checks the session. The body field is only for the demo page.
    remote = request.client.host if request.client else "anonymous"
    user_id, question, history = _accept(body, x_user_id, remote)
    result = await pipeline.run(question, user_id, history, _selected_model(body.model))
    audit.record({**result, "question": question}, user_id)
    return result


@app.post("/v1/query/stream")
async def query_stream(
    body: Query, request: Request,
    x_user_id: str | None = Header(default=None, max_length=128),
):
    remote = request.client.host if request.client else "anonymous"
    user_id, question, history = _accept(body, x_user_id, remote)
    model = _selected_model(body.model)

    async def events():
        async for event in pipeline.run_stream(question, user_id, history, model):
            if event.get("type") == "done":
                audit.record({**event, "question": question}, user_id)
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
# Keep the demo interface in a separate HTML file so its chat state can be tested directly.
PAGE = Path(__file__).with_name("chat.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
