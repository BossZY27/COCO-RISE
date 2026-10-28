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
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from swagger_ui_bundle import swagger_ui_path
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.core import security as limits
from app.infrastructure.json import audit
from app.infrastructure.postgres import inspector as db_inspector
from app.modules.knowledge import corpus as store
from app.infrastructure.ollama import legacy as llm
from app.modules.documents import pdf_inspection
from app.modules.query import engine as pipeline
from app.workers import ingestion
from app.modules.query import persona as identity
from .admin import router as admin_router
from .schemas import Feedback, Query


def _enabled(name: str, default: str = "off") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "on", "true", "yes"}


def _trusted_hosts() -> list[str]:
    raw = os.getenv("RISE_TRUSTED_HOSTS", "127.0.0.1,localhost,testserver")
    hosts = [host.strip() for host in raw.split(",") if host.strip()]
    if not hosts:
        raise RuntimeError("RISE_TRUSTED_HOSTS ต้องมี host อย่างน้อยหนึ่งค่า")
    if any("*" in host or "://" in host or "/" in host for host in hosts):
        raise RuntimeError("RISE_TRUSTED_HOSTS ต้องเป็นชื่อ host หรือ IP แบบเจาะจงและห้ามใช้ *")
    return hosts


def _trusted_proxy_ips() -> frozenset[str]:
    """Return exact proxy IPs allowed to assert an authenticated user header."""
    raw = os.getenv("RISE_TRUSTED_PROXY_IPS", "127.0.0.1,::1")
    addresses: set[str] = set()
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            addresses.add(str(ipaddress.ip_address(value)))
        except ValueError as exc:
            raise RuntimeError("RISE_TRUSTED_PROXY_IPS ต้องระบุเฉพาะ IP แบบเจาะจง") from exc
    if not addresses:
        raise RuntimeError("RISE_TRUSTED_PROXY_IPS ต้องมี IP อย่างน้อยหนึ่งค่า")
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


DOCS_ENABLED = _enabled("RISE_EXPOSE_DOCS", "on")
TRUST_PROXY_USER = _enabled("RISE_TRUST_PROXY_USER", "off")
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
            name="rise-embedding-warmup",
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
    db_inspector.sync_database()
    if pipeline.ANSWER_MODE == "llm":
        STATUS["llm"] = f"ใช้โมเดล {llm.SMALL or llm.MAIN} สร้างคำตอบ RAG จากหลักฐานที่ตรวจแล้ว"
    elif pipeline.ANSWER_MODE == "typed_claim":
        STATUS["llm"] = f"ใช้โมเดล {llm.SMALL or llm.MAIN} สร้าง Typed Claim JSON และตรวจด้วย Verifier"
    else:
        STATUS["llm"] = "ปิดการเขียนคำตอบอิสระ ใช้ข้อความจากข้อมูลที่ตรวจแล้ว"
    yield


app = FastAPI(
    title="COCO-RISE demo",
    version="0.4.0",
    description=(
        "API ของเดโม COCO-RISE ปัจจุบัน ข้อมูลโหลดจาก JSON และคำตอบสร้างจาก template "
        "โดยตรวจหลักฐานก่อนส่ง; Ollama ใช้ค้นเชิงความหมายได้เมื่อเปิดใช้งานเท่านั้น"
    ),
    openapi_tags=[
        {"name": "System", "description": "สถานะและข้อมูลประกอบของเดโม"},
        {"name": "Query", "description": "ถามข้อมูลทุนและส่งความเห็นต่อคำตอบ"},
        {"name": "Admin review", "description": "API ผู้ดูแล ต้องส่ง X-Admin-Token เมื่อเปิดใช้งาน"},
        {"name": "Admin ingestion", "description": "นำเข้า PDF เป็นฉบับร่างเพื่อให้คนตรวจและอนุมัติ"},
    ],
    docs_url=None,
    redoc_url="/redoc" if DOCS_ENABLED else None,
    openapi_url="/openapi.json" if DOCS_ENABLED else None,
    lifespan=lifespan,
)
app.mount("/_swagger", StaticFiles(directory=str(swagger_ui_path)), name="swagger-assets")
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
            or request.url.path.startswith("/v1/admin/") or request.url.path.startswith("/v2/")):
        headers["Cache-Control"] = "no-store, private"
    return response


@app.exception_handler(ingestion.IngestionStorageError)
async def ingestion_storage_error(_request: Request, exc: ingestion.IngestionStorageError):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.include_router(admin_router)


def _demo_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title="RISE API",
        version=app.version,
        openapi_version="3.0.3",
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )
    schema.setdefault("components", {}).setdefault("securitySchemes", {})["AdminToken"] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-Admin-Token",
        "description": "ใช้ token ที่ผู้ดูแลตั้งใน RISE_ADMIN_TOKEN",
    }
    schema = _openapi_30_compat(schema)
    descriptions = {
        "/healthz": "สถานะข้อมูลและบริการเสริมของเดโม; สถานะ ok ไม่ได้รับรองว่า Ollama พร้อมใช้งาน",
        "/v1/query": "คำตอบมาจาก template ของข้อมูลที่ตรวจแล้ว; persona_id เป็นผู้ใช้สมมติ",
        "/v1/query/stream": "ส่ง SSE preview/done จาก template ที่ผ่านการตรวจ ไม่ใช่ token จาก LLM",
        "/v1/admin/inspect-pdf": "ตรวจตัวอย่าง PDF โดยไม่บันทึกไฟล์หรืออนุมัติข้อความ",
        "/v1/admin/documents": "อัปโหลด PDF เข้าพื้นที่กักกัน; ทุก chunk เริ่มเป็นฉบับร่าง",
        "/v1/admin/documents/{document_id}/publish": "เผยแพร่เฉพาะข้อความที่คนยืนยัน; ไม่สร้าง facts หรือ rules",
    }
    summaries = {
        ("get", "/healthz"): "สถานะระบบ",
        ("get", "/v1/models"): "สถานะโมเดลตอบ",
        ("get", "/v1/personas"): "รายการผู้ใช้สมมติ",
        ("post", "/v1/query"): "ถามข้อมูลทุน",
        ("post", "/v1/query/stream"): "ถามข้อมูลทุนแบบ SSE",
        ("post", "/v1/feedback"): "ส่งความเห็นต่อคำตอบ",
        ("post", "/v1/admin/inspect-pdf"): "ตรวจตัวอย่าง PDF โดยไม่บันทึก",
        ("post", "/v1/admin/documents"): "อัปโหลด PDF เข้าพื้นที่รอตรวจ",
        ("get", "/v1/admin/documents"): "แสดงรายการเอกสารรอตรวจ",
    }
    schema_components = schema.setdefault("components", {}).setdefault("schemas", {})
    schema_components["DemoQueryResponse"] = {
        "type": "object",
        "description": "ผล query ของเดโมปัจจุบัน; answer_source เป็น template หรือ blocked",
        "properties": {
            "answer": {"type": "string"}, "citations": {"type": "array", "items": {"type": "string"}},
            "slots": {"type": "object", "additionalProperties": True},
            "ruling": {"type": "object", "additionalProperties": True},
            "answer_source": {"type": "string", "enum": ["template", "blocked"]},
            "selected_model": {"type": "string", "nullable": True},
            "evidence": {"type": "object", "additionalProperties": True},
            "verify": {"type": "object", "properties": {"ok": {"type": "boolean"},
                "problems": {"type": "array", "items": {"type": "string"}}}, "required": ["ok", "problems"]},
            "timings_ms": {"type": "object", "additionalProperties": {"type": "number"}},
            "profile": {"type": "object", "additionalProperties": True},
        },
        "required": ["answer", "citations", "slots", "ruling", "answer_source", "selected_model",
                     "evidence", "verify", "timings_ms", "profile"],
    }
    schema_components["DemoFeedbackResponse"] = {
        "type": "object", "properties": {"ok": {"type": "boolean"}, "recorded": {"type": "boolean"}},
        "required": ["ok", "recorded"],
    }
    schema.setdefault("components", {}).setdefault("responses", {}).update({
        "InvalidRequest": {"description": "request ไม่ผ่าน validation หรือเงื่อนไข", "content":
            {"application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}}},
        "AdminUnauthorized": {"description": "ไม่มีหรือ token ผู้ดูแลไม่ถูกต้อง", "content":
            {"application/json": {"schema": {"type": "object", "properties": {"detail": {"type": "string"}}}}}},
        "ResourceNotFound": {"description": "ไม่พบเอกสาร งาน หรือ chunk", "content":
            {"application/json": {"schema": {"type": "object", "properties": {"detail": {"type": "string"}}}}}},
        "Conflict": {"description": "สถานะปัจจุบันไม่อนุญาตให้ทำรายการ", "content":
            {"application/json": {"schema": {"type": "object", "properties": {"detail": {"type": "string"}}}}}},
        "RateLimited": {"description": "ส่งคำขอถี่เกินกำหนด", "content":
            {"application/json": {"schema": {"type": "object", "properties": {"detail": {"type": "string"}}}}}},
    })
    admin_prefixes = ("/v1/admin/", "/v1/recent")
    for path, path_item in schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if not isinstance(operation, dict):
                continue
            operation.setdefault("description", descriptions.get(path, "RISE API."))
            operation["summary"] = summaries.get((method, path), operation.get("summary", "COCO-RISE API"))
            if path in {"/healthz", "/v1/models", "/v1/personas"}:
                operation["tags"] = ["System"]
            elif path.startswith(admin_prefixes):
                operation["tags"] = ["Admin ingestion" if any(x in path for x in
                    ("inspect-pdf", "/documents", "/jobs")) else "Admin review"]
                operation["security"] = [{"AdminToken": []}]
                operation["parameters"] = [p for p in operation.get("parameters", [])
                                            if not (p.get("in") == "header" and
                                                    p.get("name", "").lower() == "x-admin-token")]
            elif path.startswith("/v1/"):
                operation["tags"] = ["Query"]
            responses = operation.setdefault("responses", {})
            for code, ref in (("422", "InvalidRequest"), ("429", "RateLimited")):
                responses.setdefault(code, {"$ref": f"#/components/responses/{ref}"})
            if path.startswith(admin_prefixes) or (path.startswith("/v2/") and any(t in operation.get("tags", []) for t in ["Documents v2", "Knowledge v2", "Jobs v2"]) and not path.startswith("/v2/sources/")):
                operation["security"] = [{"AdminToken": []}]
                responses.setdefault("401", {"$ref": "#/components/responses/AdminUnauthorized"})
                if method == "get" or "documents" in path or "jobs" in path:
                    responses.setdefault("404", {"$ref": "#/components/responses/ResourceNotFound"})
                responses.setdefault("409", {"$ref": "#/components/responses/Conflict"})
            if path == "/v1/query" and method == "post":
                responses["200"] = {"description": "ผลคำตอบจากข้อมูลที่ตรวจแล้ว",
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/DemoQueryResponse"},
                    "example": {"answer": "ยังสรุปสิทธิ์ไม่ได้ เพราะกฎที่ตรวจแล้วไม่เพียงพอ", "citations": [],
                        "slots": {"intent": "eligibility"}, "ruling": {"verdict": "no_rule"},
                        "answer_source": "template", "selected_model": None,
                        "evidence": {"chunks": [], "facts": [], "keyword": [], "semantic": [], "semantic_by": "trigram"},
                        "verify": {"ok": True, "problems": []}, "timings_ms": {"analyze": 1.0, "retrieve": 2.0},
                        "profile": {"id": "P1", "name": "ผู้ใช้สมมติ 1", "source": "demo"}}}}}
            elif path == "/v1/query/stream" and method == "post":
                responses["200"] = {"description": "Server-Sent Events: preview (ถ้ามี) และ done/blocked",
                    "content": {"text/event-stream": {"schema": {"type": "string"},
                        "example": "data: {\"type\":\"done\",\"answer\":\"คำตอบที่ตรวจแล้ว\"}\n\n"}}}
            elif path == "/v1/feedback" and method == "post":
                responses["200"] = {"description": "รับความเห็นแล้ว", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/DemoFeedbackResponse"},
                    "example": {"ok": True, "recorded": False}}}}
            elif path in {"/v1/admin/inspect-pdf", "/v1/admin/documents"} and method == "post":
                operation["requestBody"] = {"required": True, "content": {"application/pdf": {
                    "schema": {"type": "string", "format": "binary"},
                    "example": "ส่งไบต์ของไฟล์ PDF โดยตรง"}}}
                responses.setdefault("400", {"description": "PDF อ่านไม่ได้หรือไม่ถูกต้อง"})
                responses.setdefault("413", {"description": "ไฟล์ใหญ่เกิน 15 MiB"})
                responses.setdefault("415", {"description": "ต้องส่ง Content-Type: application/pdf"})
    app.openapi_schema = schema
    return schema


def _openapi_30_compat(value):
    """Convert nullable JSON Schema unions emitted by Pydantic into OpenAPI 3.0 form.

    The bundled Swagger UI is Swagger UI 4.x, which accepts OpenAPI 3.0.x but
    rejects 3.1.x. Keep unions such as string-or-integer as oneOf.
    """
    if isinstance(value, list):
        return [_openapi_30_compat(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _openapi_30_compat(item) for key, item in value.items() if key != "anyOf"}
    alternatives = value.get("anyOf")
    if alternatives:
        null_branch = any(isinstance(item, dict) and item.get("type") == "null" for item in alternatives)
        non_null = [_openapi_30_compat(item) for item in alternatives
                    if not (isinstance(item, dict) and item.get("type") == "null")]
        if null_branch and len(non_null) == 1:
            result.update(non_null[0])
            result["nullable"] = True
        elif null_branch:
            result["oneOf"] = non_null
            result["nullable"] = True
        else:
            result["oneOf"] = non_null
    declared_type = result.get("type")
    if isinstance(declared_type, list) and "null" in declared_type:
        other_types = [item for item in declared_type if item != "null"]
        if len(other_types) == 1:
            result["type"] = other_types[0]
            result["nullable"] = True
    if isinstance(result.get("enum"), list) and None in result["enum"]:
        result["enum"] = [item for item in result["enum"] if item is not None]
        result["nullable"] = True
    return result


app.openapi = _demo_openapi


if DOCS_ENABLED:
    @app.get("/docs", include_in_schema=False)
    async def demo_docs():
        """Swagger UI for the running demo API, with assets served locally."""
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} — Swagger UI",
            swagger_js_url="/_swagger/swagger-ui-bundle.js",
            swagger_css_url="/_swagger/swagger-ui.css",
            swagger_favicon_url="/_swagger/favicon-32x32.png",
            oauth2_redirect_url="/docs/oauth2-redirect",
            swagger_ui_parameters={"displayRequestDuration": True, "filter": True},
        )

    @app.get("/design/openapi.json", include_in_schema=False)
    async def design_openapi():
        """Read-only OpenAPI contract generated from implemented v2 schemas."""
        design_path = Path(__file__).resolve().parents[3] / "docs" / "openapi-mvp.json"
        return JSONResponse(json.loads(design_path.read_text(encoding="utf-8")))

    @app.get("/design/docs", include_in_schema=False)
    async def design_docs():
        """Read-only Swagger UI for the proposed MVP API."""
        return get_swagger_ui_html(
            openapi_url="/design/openapi.json",
            title="COCO-RISE MVP Design — Swagger UI",
            swagger_js_url="/_swagger/swagger-ui-bundle.js",
            swagger_css_url="/_swagger/swagger-ui.css",
            swagger_favicon_url="/_swagger/favicon-32x32.png",
            swagger_ui_parameters={
                "supportedSubmitMethods": [],
                "displayRequestDuration": True,
                "filter": True,
            },
        )


def _accept(
    body: Query, header_user: str | None, client_host: str
) -> tuple[str, str, list[str]]:
    """Identity, input limits and the rate limit, applied the same way on both query endpoints."""
    trusted_header = _proxy_user_id(header_user, client_host)
    if not trusted_header and body.persona_id not in store.view().PERSONAS:
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
    unverified = sum(1 for f in store.view().FACTS if not store.is_verified(f))
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None and not runtime.database.ready():
        raise HTTPException(status_code=503, detail="Migrations not current")
    main_model = (llm.SMALL or llm.MAIN) if pipeline.ANSWER_MODE in ("llm", "typed_claim") else None
    return {"status": "ok", "chunks": len(store.view().CHUNKS), "facts": len(store.view().FACTS), "rules": len(store.view().RULES),
            "facts_awaiting_review": unverified,
            "answer_mode": pipeline.ANSWER_MODE, "main_model": main_model, "embed_model": llm.EMBED_MODEL,
            "vectors_ready": store.vectors_ready(), "semantic_backend": pipeline.semantic_backend(),
            "warmup": STATUS}


@app.get("/v1/models")
def models():
    if pipeline.ANSWER_MODE == "llm":
        available = llm.answer_models()
        default_model = llm.SMALL or llm.MAIN
        return {
            "enabled": True,
            "default": default_model,
            "models": available,
            "answer_mode": pipeline.ANSWER_MODE,
            "message": f"โมเดล {default_model} สร้างคำตอบ RAG จากหลักฐานที่ตรวจแล้ว พร้อมตรวจตัวเลขและเอกสารอ้างอิงทุกครั้ง",
        }
    if pipeline.ANSWER_MODE == "typed_claim":
        available = llm.answer_models()
        default_model = llm.SMALL or llm.MAIN
        return {
            "enabled": True,
            "default": default_model,
            "models": available,
            "answer_mode": pipeline.ANSWER_MODE,
            "message": f"โมเดล {default_model} สร้างเฉพาะ Typed Claim JSON (อ้างอิง fact_id/chunk_id) และผ่าน Verifier ทุกครั้ง",
        }
    return {"enabled": False, "default": None, "models": [],
            "answer_mode": pipeline.ANSWER_MODE,
            "message": "โหมดปลอดภัยใช้คำตอบที่โค้ดประกอบจากข้อมูลที่ตรวจแล้ว"}


def _selected_model(name: str | None) -> str | None:
    if not name:
        return None
    if pipeline.ANSWER_MODE not in ("llm", "typed_claim"):
        raise HTTPException(status_code=400, detail="โหมดปลอดภัยยังไม่เปิดให้โมเดลเขียนคำตอบอิสระ")
    allowed = {m["name"] for m in llm.answer_models()} | {llm.SMALL, llm.MAIN, "template"}
    if name not in allowed:
        raise HTTPException(status_code=400, detail="ไม่พบโมเดลที่เลือกในรายการที่ระบบอนุญาต")
    return name


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
    from app.core.runtime import services as runtime_services
    recorded = bool(audit.ENABLED or runtime_services.get() is not None)
    if recorded:
        audit.record({"question": limits.clean_feedback_text(body.question),
                      "answer": limits.clean_feedback_text(body.answer),
                      "answer_source": "feedback", "citations": [],
                      "ruling": {"verdict": "helpful" if body.helpful else "not_helpful"},
                      "slots": {"note": limits.clean_feedback_text(body.note)},
                      "timings_ms": {}}, user_id)
    return {"ok": True, "recorded": recorded}




@app.get("/v1/personas")
def personas():
    # PDPA Data Minimization: expose only ID, display label, and source to the browser.
    return [
        {"id": p["id"], "name": p["name"], "source": p.get("source", "unknown")}
        for p in store.view().PERSONAS.values()
    ]


@app.post("/api/query", summary="Ask a verified research-fund question")
@app.post("/v1/query", summary="Ask a verified research-fund question")
async def query(
    body: Query, request: Request,
    x_user_id: str | None = Header(default=None, max_length=128),
):
    # The proxy sets X-User-Id after it checks the session. The body field is only for the demo page.
    remote = request.client.host if request.client else "anonymous"
    user_id, question, history = _accept(body, x_user_id, remote)
    context = body.context.model_dump() if body.context else None
    result = await pipeline.run(question, user_id, history, _selected_model(body.model), context)
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
    context = body.context.model_dump() if body.context else None

    async def events():
        async for event in pipeline.run_stream(question, user_id, history, model, context):
            if event.get("type") == "done":
                audit.record({**event, "question": question}, user_id)
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
# Keep the demo interface in a separate HTML file so its chat state can be tested directly.
PAGE = (Path(__file__).resolve().parents[2] / "static" / "chat.html").read_text(encoding="utf-8")
DB_PAGE = (Path(__file__).resolve().parents[2] / "static" / "db_viewer.html").read_text(encoding="utf-8")


@app.get("/v1/db/overview", summary="Inspect the 4-layer SQLite/PostgreSQL database")
def db_overview(table: str = "document_registry", only_verified: bool = False, q: str = ""):
    return db_inspector.get_overview(table=table, only_verified=only_verified, q=q)


@app.get("/db", response_class=HTMLResponse, include_in_schema=False)
def db_viewer():
    return DB_PAGE


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
