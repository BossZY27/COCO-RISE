"""RISE MVP PostgreSQL schema (paired with ChromaDB vector index)."""
from alembic import op
from app.infrastructure.postgres.models import Base

revision = "0001_rise"
down_revision = None


def upgrade():
    for name in ("core", "knowledge", "retrieval", "audit"):
        op.execute(f"CREATE SCHEMA IF NOT EXISTS {name}")
    Base.metadata.create_all(op.get_bind())
    op.execute("INSERT INTO retrieval.corpus_revision (id, revision) VALUES (1, 0)")


def downgrade():
    Base.metadata.drop_all(op.get_bind())
    for name in ("audit", "retrieval", "knowledge", "core"):
        op.execute(f"DROP SCHEMA {name}")
