"""Vector repository facade backed by ChromaDB and PostgreSQL revision metadata."""

from app.infrastructure.chroma.vectors import ChromaVectorRepository as VectorRepository

__all__ = ["VectorRepository"]
