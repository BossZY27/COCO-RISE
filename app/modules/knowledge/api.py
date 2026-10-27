"""HTTP request validation and application service delegation."""
from uuid import UUID
from fastapi import APIRouter, HTTPException, Request
from app.core.dependencies import Admin, Limit, Offset, services
from app.core import security as limits
from app.modules.query.schemas import FeedbackInput, QueryInput, QueryResponse
from app.modules.knowledge.schemas import Decision, KnowledgeInput, KnowledgeResponse, Record

router = APIRouter(prefix="/v2")

def knowledge_routes(kind, path):
    def listing(request: Request, _: Admin, limit: Limit = 50, offset: Offset = 0):
        return services(request).repository.list_knowledge(kind, limit, offset)

    def get(item_id: UUID, request: Request, _: Admin):
        return services(request).repository.get_knowledge(kind, item_id)

    def create(body: KnowledgeInput, request: Request, _: Admin):
        return services(request).repository.save_knowledge(kind, body)

    def edit(item_id: UUID, body: KnowledgeInput, request: Request, _: Admin):
        return services(request).repository.save_knowledge(kind, body, item_id)

    def decide(item_id: UUID, body: Decision, request: Request, _: Admin):
        return services(request).repository.decide_knowledge(kind, item_id, body)

    for suffix, endpoint, method, response_model, code in [
        ("", listing, "GET", list[KnowledgeResponse], 200), ("/{item_id}", get, "GET", KnowledgeResponse, 200),
        ("", create, "POST", KnowledgeResponse, 201), ("/{item_id}", edit, "PATCH", KnowledgeResponse, 200),
        ("/{item_id}/review", decide, "POST", KnowledgeResponse, 200)]:
        router.add_api_route(path + suffix, endpoint, methods=[method], response_model=response_model,
                             status_code=code, tags=["Knowledge v2"], name=kind + "_" + endpoint.__name__)


knowledge_routes("fact", "/facts")
knowledge_routes("rule", "/rules")
