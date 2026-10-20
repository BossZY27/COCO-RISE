"""HTTP request validation and application service delegation."""
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from app.core.dependencies import Admin, Limit, Offset, services
from app.core import security as limits
from app.modules.query.schemas import FeedbackInput, QueryInput, QueryResponse
from app.modules.knowledge.schemas import Decision, KnowledgeInput, KnowledgeResponse, Record

router = APIRouter(prefix="/v2")

@router.post("/query", response_model=QueryResponse, tags=["Query v2"])
def query(body: QueryInput, request: Request):
    app = services(request)
    remote = request.client.host if request.client else "anonymous"
    allowed, wait = limits.take_client(remote)
    if not allowed:
        raise HTTPException(429, "Query rate limit exceeded", headers={"Retry-After": str(wait)})
    question, refusal = limits.clean_question(body.question)
    if refusal:
        raise HTTPException(422, refusal)
    return app.query.run(question, limits.clean_history(body.history))

