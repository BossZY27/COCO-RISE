"""Public readiness and authenticated operational views of the real database."""
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select, func
from app.core.dependencies import Admin, Limit, services
from app.infrastructure.postgres.models import Document, Version, Page, Chunk, Knowledge, QueryRun, Job, Generation
from app.infrastructure.postgres.repository import record

router = APIRouter(prefix="/api")


@router.get("/status", tags=["System"])
def status(request: Request):
    return services(request).status()


@router.get("/ready", tags=["System"])
def ready(request: Request):
    if not services(request).database.ready():
        raise HTTPException(503, "Migrations not current")
    return {"ready": True}


@router.get("/recent", tags=["System"])
def recent(request: Request, _: Admin, limit: Limit = 50):
    with services(request).database.transaction() as s:
        return [record(r) for r in s.scalars(select(QueryRun).order_by(QueryRun.created_at.desc()).limit(limit))]


@router.get("/db/overview", tags=["System"])
def overview(request: Request, _: Admin, table: str = "documents", limit: Limit = 50):
    tables = {m.__tablename__: m for m in (Document, Version, Page, Chunk, Knowledge, QueryRun, Job, Generation)}
    if table not in tables:
        raise HTTPException(422, "Unknown table")
    with services(request).database.transaction() as s:
        return {"engine": "PostgreSQL + ChromaDB", "selected_table": table,
                "counts": {name: s.scalar(select(func.count()).select_from(model)) for name, model in tables.items()},
                "rows": [record(r) for r in s.scalars(select(tables[table]).limit(limit))]}
