from typing import Protocol


class AnswerModel(Protocol):
    def candidate(self, template: str, citations: list[str]) -> tuple[str, str]: ...


class Retrieval(Protocol):
    def search(self, question: str, year: int | None, corpus) -> tuple[list[dict], str]: ...
