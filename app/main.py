"""Application assembly; v1 compatibility and v2 share lifecycle-bound adapters."""
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

import sys
if "pytest" not in sys.modules:
    load_dotenv(override=False)

from .modules.legacy import api as compat
from .bootstrap import Services
from .core.errors import ServiceError
from .core.runtime import services as runtime_services
from .core.settings import Settings
from .infrastructure.postgres.compat import LegacyAdapter
from .modules.documents.api import router as documents_router
from .modules.query.api import router as query_router
from .modules.knowledge.api import router as knowledge_router
from .modules.feedback.api import router as feedback_router
from .modules.system.api import router as system_router
from .modules.knowledge.context import use_corpus


@asynccontextmanager
async def lifespan(app):
    settings = Settings()
    app.state.settings = settings
    app.state.services = Services(settings) if settings.database_url else None
    if app.state.services is not None:
        app.state.services.legacy = LegacyAdapter(app.state.services)
        try:
            yield
        finally:
            app.state.services.close()
            app.state.services = None
    else:
        async with compat.lifespan(app):
            yield


class RuntimeContext:
    def __init__(self, app, application):
        self.app, self.application = app, application

    async def __call__(self, scope, receive, send):
        runtime = getattr(self.application.state, "services", None)
        if scope["type"] != "http" or runtime is None:
            await self.app(scope, receive, send)
            return
        token = runtime_services.set(runtime)
        try:
            if scope["path"].startswith("/v1/") or scope["path"] in {"/healthz", "/api/query"}:
                try:
                    corpus = await asyncio.to_thread(runtime.corpora.load)
                except SQLAlchemyError:
                    await JSONResponse({"detail": "Database unavailable or migrations required"}, status_code=503)(scope, receive, send)
                    return
                with use_corpus(corpus):
                    await self.app(scope, receive, send)
            else:
                await self.app(scope, receive, send)
        finally:
            runtime_services.reset(token)


app = compat.app
app.title, app.version = "RISE API", "2.0.0"
app.description = "RISE PostgreSQL MVP: versioned evidence, human review, hybrid retrieval, verified answers. /v1 retains the demo contract."
app.router.lifespan_context = lifespan
for router in (documents_router, query_router, knowledge_router, feedback_router, system_router):
    app.include_router(router)
app.add_middleware(RuntimeContext, application=app)


@app.exception_handler(ServiceError)
async def service_error(_request, exc):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status, headers={"Cache-Control": "no-store"})


@app.exception_handler(SQLAlchemyError)
async def database_error(_request, _exc):
    return JSONResponse({"detail": "Database unavailable or conflicting operation"}, status_code=503,
                        headers={"Cache-Control": "no-store"})
