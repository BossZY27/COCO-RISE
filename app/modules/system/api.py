"""HTTP request validation and application service delegation."""
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from app.core.dependencies import Admin, Limit, Offset, services
from app.core import security as limits
from app.modules.query.schemas import FeedbackInput, QueryInput, QueryResponse
from app.modules.knowledge.schemas import Decision, KnowledgeInput, KnowledgeResponse, Record

router = APIRouter(prefix="/v2")

@router.get("/status", tags=["System v2"])
def status(request: Request):
    return services(request).status()


@router.get("/ready", tags=["System v2"])
def ready(request: Request):
    if not services(request).database.ready():
        raise HTTPException(503, "Migrations not current")
    return {"ready": True}

