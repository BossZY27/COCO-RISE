"""Filesystem adapter for the human-reviewed PDF publication workflow.

The quarantine layout belongs to ingestion.  Keeping its private file helpers
here prevents the publication policy from depending on that layout directly.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.modules.knowledge import corpus as store
from app.workers import ingestion


class LocalPublicationRepository:
    def __init__(self, published_path: Path):
        self.published_path = published_path

    @contextmanager
    def transaction(self) -> Iterator[None]:
        # Ingestion updates and publication must not interleave in this process.
        with ingestion._LOCK:
            yield

    def valid_document_id(self, document_id: str) -> bool:
        return ingestion._document_dir(document_id) is not None

    def read_document(self, document_id: str):
        directory = ingestion._document_dir(document_id)
        return ingestion._read_json(directory / "document.json") if directory else None

    def read_chunks(self, document_id: str):
        directory = ingestion._document_dir(document_id)
        return ingestion._read_json(directory / "chunks.json") if directory else None

    def source_bytes(self, document_id: str) -> bytes | None:
        source = ingestion.source_path(document_id)
        return source.read_bytes() if source else None

    def read_published(self):
        return ingestion._read_json(self.published_path)

    def write_review(self, document_id: str, chunks: list[dict], document: dict) -> None:
        directory = ingestion._document_dir(document_id)
        if directory is None:
            raise ValueError("รหัสเอกสารไม่ถูกต้อง")
        ingestion._atomic_json(directory / "chunks.json", chunks)
        ingestion._atomic_json(directory / "document.json", document)

    def publish(self, chunks: list[dict]) -> None:
        """Replace the live corpus last; a failed earlier write stays quarantined."""
        ingestion._atomic_json(self.published_path, chunks)
        store.reload_published_chunks(self.published_path)
        store.refresh_derived_indexes()

    def now(self) -> str:
        return ingestion._now()
