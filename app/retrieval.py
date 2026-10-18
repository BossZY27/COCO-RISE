"""Approved-only, document-aware passage retrieval and comparison.

The index ranks evidence but never approves a document or composes an answer.
It can be rebuilt from a fresh chunk snapshot after a human review step.
"""

from collections import Counter, defaultdict
from math import sqrt
import re

from rank_bm25 import BM25Okapi

from .thai import tokenize, trigrams


def _cosine(a: Counter, b: Counter) -> float:
    shared = set(a) & set(b)
    dot = sum(a[key] * b[key] for key in shared)
    length_a = sqrt(sum(value * value for value in a.values()))
    length_b = sqrt(sum(value * value for value in b.values()))
    return dot / (length_a * length_b) if length_a and length_b else 0.0


class ApprovedEvidenceIndex:
    def __init__(self, chunks: list[dict], vocabulary: set[str], fiscal_year: int | None = None):
        self.vocabulary = vocabulary
        self.chunks = [chunk for chunk in chunks
                       if chunk.get("status") == "active" and chunk.get("verified") is True
                       and (fiscal_year is None or chunk.get("fiscal_year") in (None, fiscal_year))]
        self.child_tokens = [tokenize(chunk["text"], vocabulary) for chunk in self.chunks]
        self.child_grams = [Counter(trigrams(chunk["text"])) for chunk in self.chunks]
        self.child_bm25 = BM25Okapi(self.child_tokens) if self.chunks else None

        # Expand a child only with the heading of an approved parent clause.
        # The returned evidence remains the original child, never the synthetic text.
        def document_key(chunk: dict) -> tuple:
            # Two uploads may have the same display name.  Never borrow a
            # heading (or page) from another PDF or another annual edition.
            if chunk.get("document_id"):
                return ("id", chunk["document_id"])
            return ("label", chunk.get("doc"), chunk.get("fiscal_year"))

        first_clause = {}
        for index, chunk in enumerate(self.chunks):
            first_clause.setdefault((document_key(chunk), chunk.get("clause")), index)
        expanded_texts = []
        for index, chunk in enumerate(self.chunks):
            clause = chunk.get("clause", "")
            parent_clause = chunk.get("parent_clause")
            if not parent_clause and re.fullmatch(r"\d+(?:\.\d+){2,}", clause):
                parent_clause = clause.rsplit(".", 1)[0]
            elif not parent_clause and re.fullmatch(r"\d+\.\d+", clause):
                # A continued row may have the same clause on a later chunk.
                parent_clause = clause
            parent_index = first_clause.get((document_key(chunk), parent_clause)) if parent_clause else None
            heading = ""
            if parent_index is not None and parent_index != index:
                parent_text = self.chunks[parent_index]["text"]
                parent_text = re.sub(r"^\s*\d+(?:\.\d+)*\s*", "", parent_text)
                heading = re.split(r"\s+(?:วงเงิน|ไม่เกิน)", parent_text, maxsplit=1)[0][:120]
            expanded_texts.append(f"{heading} {chunk['text']}" if heading else chunk["text"])
        self.hier_tokens = [tokenize(text, vocabulary) for text in expanded_texts]
        self.hier_grams = [Counter(trigrams(text)) for text in expanded_texts]
        self.hier_bm25 = BM25Okapi(self.hier_tokens) if self.chunks else None

        page_children = defaultdict(list)
        for index, chunk in enumerate(self.chunks):
            page_children[(document_key(chunk), chunk.get("page"))].append(index)
        self.page_children = list(page_children.values())
        page_texts = [" ".join(self.chunks[index]["text"] for index in children)
                      for children in self.page_children]
        self.page_tokens = [tokenize(text, vocabulary) for text in page_texts]
        self.page_grams = [Counter(trigrams(text)) for text in page_texts]
        self.page_bm25 = BM25Okapi(self.page_tokens) if page_texts else None

    def _rank(self, query: str, candidates: list[int], bm25, grams: list[Counter]) -> list[int]:
        if not candidates or bm25 is None:
            return []
        keyword_scores = bm25.get_scores(tokenize(query, self.vocabulary))
        query_grams = Counter(trigrams(query))
        semantic_scores = {index: _cosine(query_grams, grams[index]) for index in candidates}
        keyword = sorted((index for index in candidates if keyword_scores[index] > 0),
                         key=lambda index: (-keyword_scores[index], index))
        semantic = sorted((index for index in candidates if semantic_scores[index] > 0.02),
                          key=lambda index: (-semantic_scores[index], index))
        fused = defaultdict(float)
        for ranking in (keyword, semantic):
            for position, index in enumerate(ranking, start=1):
                fused[index] += 1 / (60 + position)
        return sorted(fused, key=lambda index: (-fused[index], index))

    def flat(self, query: str, top_k: int = 5) -> list[dict]:
        indices = self._rank(query, list(range(len(self.chunks))),
                             self.child_bm25, self.child_grams)
        return [self.chunks[index] for index in indices[:top_k]]

    def parent_context(self, query: str, top_k: int = 5) -> list[dict]:
        indices = self._rank(query, list(range(len(self.chunks))),
                             self.hier_bm25, self.hier_grams)
        return [self.chunks[index] for index in indices[:top_k]]

    def page_only(self, query: str, top_k: int = 5, top_pages: int = 3) -> list[dict]:
        pages = self._rank(query, list(range(len(self.page_children))),
                           self.page_bm25, self.page_grams)[:top_pages]
        candidates = [index for page in pages for index in self.page_children[page]]
        indices = self._rank(query, candidates, self.child_bm25, self.child_grams)
        return [self.chunks[index] for index in indices[:top_k]]

    def hierarchical(self, query: str, top_k: int = 5, top_pages: int = 3) -> list[dict]:
        pages = self._rank(query, list(range(len(self.page_children))),
                           self.page_bm25, self.page_grams)[:top_pages]
        candidates = [index for page in pages for index in self.page_children[page]]
        indices = self._rank(query, candidates, self.hier_bm25, self.hier_grams)
        return [self.chunks[index] for index in indices[:top_k]]
