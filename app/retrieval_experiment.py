"""Backward-compatible name used by the offline retrieval benchmark."""

from . import store
from .retrieval import ApprovedEvidenceIndex


class RetrievalComparison(ApprovedEvidenceIndex):
    def __init__(self, chunks: list[dict], fiscal_year: int | None = None):
        super().__init__(chunks, store.VOCAB, fiscal_year)
