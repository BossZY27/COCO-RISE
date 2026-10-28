"""One application, one API, and lifecycle-bound PostgreSQL/model adapters."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=False)

from fastapi import FastAPI, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.trustedhost import TrustedHostMiddleware
from swagger_ui_bundle import swagger_ui_path

from app.bootstrap import Services
from app.core import security as limits
from app.core.errors import ServiceError
from app.core.settings import Settings
from app.modules.documents.api import router as documents_router
from app.modules.feedback.api import router as feedback_router
from app.modules.knowledge.api import router as knowledge_router
from app.modules.query.api import router as query_router
from app.modules.system.api import router as system_router


@asynccontextmanager
async def lifespan(app):
    settings = Settings()
    app.state.settings = settings
    app.state.services = Services(settings) if settings.database_url else None
    try:
        yield
    finally:
        if app.state.services is not None:
            app.state.services.close()
            app.state.services = None


def trusted_hosts():
    hosts = [h.strip() for h in os.getenv("RISE_TRUSTED_HOSTS", "127.0.0.1,localhost,testserver").split(",") if h.strip()]
    if not hosts or any("*" in h or "://" in h or "/" in h for h in hosts):
        raise RuntimeError("RISE_TRUSTED_HOSTS requires exact hosts or IP addresses")
    return hosts


DOCS_ENABLED = os.getenv("RISE_EXPOSE_DOCS", "on").lower() in {"on", "true", "1", "yes"}
app = FastAPI(title="RISE API", version="3.0.0", lifespan=lifespan,
              description="Grounded grant QA with reviewed PDF evidence, PostgreSQL, ChromaDB and Ollama.",
              docs_url=None, redoc_url=None, openapi_url="/openapi.json" if DOCS_ENABLED else None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts(), www_redirect=False)
app.add_middleware(limits.RequestBodyLimitMiddleware,
                   path_limits={"/api/documents/inspect-pdf": 15 * 1024 * 1024})
for router in (documents_router, query_router, knowledge_router, feedback_router, system_router):
    app.include_router(router)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    for name, value in {
        "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer", "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Resource-Policy": "same-origin",
        "Content-Security-Policy": "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; form-action 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'",
    }.items():
        response.headers.setdefault(name, value)
    if request.url.path.startswith("/api/") or request.url.path in {"/admin", "/db"}:
        response.headers["Cache-Control"] = "no-store, private"
    return response


@app.exception_handler(ServiceError)
async def service_error(_request, exc):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status, headers={"Cache-Control": "no-store"})


@app.exception_handler(SQLAlchemyError)
async def database_error(_request, _exc):
    return JSONResponse({"detail": "Database unavailable or migrations required"}, status_code=503,
                        headers={"Cache-Control": "no-store"})


STATIC = Path(__file__).parent / "static"

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def chat():
    return (STATIC / "chat.html").read_text(encoding="utf-8")


@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def admin_page():
    return (STATIC / "admin.html").read_text(encoding="utf-8")


@app.get("/db", response_class=HTMLResponse, include_in_schema=False)
def database_page():
    return (STATIC / "db_viewer.html").read_text(encoding="utf-8")


if DOCS_ENABLED:
    app.mount("/_swagger", StaticFiles(directory=str(swagger_ui_path)), name="swagger-assets")

    @app.get("/docs", include_in_schema=False)
    def swagger():
        return get_swagger_ui_html(openapi_url="/openapi.json", title="RISE API",
                                  swagger_js_url="/_swagger/swagger-ui-bundle.js",
                                  swagger_css_url="/_swagger/swagger-ui.css",
                                  swagger_favicon_url="/_swagger/favicon-32x32.png")
