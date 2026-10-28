from typing import Protocol


class AnswerModel(Protocol):
    def candidate(self, template: str, citations: list[str], timeout: float | None = None) -> tuple[str, str]: ...


class Retrieval(Protocol):
    def search(self, question: str, year: int | None, corpus, timeout: float | None = None) -> tuple[list[dict], str, str | None]: ...
