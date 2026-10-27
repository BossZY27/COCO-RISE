"""HTTP request validation and application service delegation."""
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from app.core.dependencies import Admin, Limit, Offset, services
from app.core import security as limits
from app.modules.query.schemas import FeedbackInput, QueryInput, QueryResponse
from app.modules.knowledge.schemas import Decision, KnowledgeInput, KnowledgeResponse, Record

router = APIRouter(prefix="/v2")

@router.post("/feedback", response_model=Record, status_code=201, tags=["Feedback v2"])
def feedback(body: FeedbackInput, request: Request):
    remote = request.client.host if request.client else "anonymous"
    allowed, wait = limits.take_feedback_client(remote)
    if not allowed:
        raise HTTPException(429, "Feedback rate limit exceeded", headers={"Retry-After": str(wait)})
    return services(request).repository.add_feedback(body)

