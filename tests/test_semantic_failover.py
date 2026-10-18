"""Regression coverage for optional semantic retrieval outages."""

from app import llm, pipeline, store


def test_embedding_outage_uses_a_circuit_breaker_instead_of_waiting_every_query(monkeypatch):
    calls = []
    fallback = [{"id": "trigram-hit"}]

    monkeypatch.setattr(store, "vectors_ready", lambda: True)
    monkeypatch.setattr(store, "semantic_search", lambda *_args: fallback)
    monkeypatch.setattr(pipeline, "_SEMANTIC_FAILURE_UNTIL", 0.0, raising=False)

    def unavailable(_texts, timeout):
        calls.append(timeout)
        raise TimeoutError("embedding service unavailable")

    monkeypatch.setattr(llm, "embed", unavailable)

    assert pipeline._semantic("ทุนวิจัย", 2569) == (fallback, "trigram")
    assert pipeline._semantic("ทุนวิจัย", 2569) == (fallback, "trigram")
    assert len(calls) == 1
    assert calls[0] <= 3.0
    assert pipeline.semantic_backend() == "trigram"


def test_embedding_circuit_retries_after_cooldown(monkeypatch):
    clock = [100.0]
    attempts = []
    vector_hits = [{"id": "vector-hit"}]

    monkeypatch.setattr(store, "vectors_ready", lambda: True)
    monkeypatch.setattr(store, "semantic_search", lambda *_args: [])
    monkeypatch.setattr(store, "vector_search", lambda *_args: vector_hits)
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(pipeline, "_SEMANTIC_FAILURE_UNTIL", 0.0, raising=False)

    def fail_once(_texts, timeout):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise TimeoutError("temporary outage")
        return [[0.1, 0.2]]

    monkeypatch.setattr(llm, "embed", fail_once)

    assert pipeline._semantic("ทุนวิจัย", 2569)[1] == "trigram"
    clock[0] += pipeline.SEMANTIC_RETRY_SECONDS - 0.1
    assert pipeline._semantic("ทุนวิจัย", 2569)[1] == "trigram"
    assert len(attempts) == 1
    clock[0] += 0.2
    assert pipeline._semantic("ทุนวิจัย", 2569) == (vector_hits, llm.EMBED_MODEL)
    assert len(attempts) == 2


def test_semantic_result_reports_the_configured_embedding_model(monkeypatch):
    vector_hits = [{"id": "vector-hit"}]

    monkeypatch.setattr(store, "vectors_ready", lambda: True)
    monkeypatch.setattr(store, "vector_search", lambda *_args: vector_hits)
    monkeypatch.setattr(llm, "EMBED_MODEL", "alternate-embed:model")
    monkeypatch.setattr(llm, "embed", lambda _texts, timeout: [[0.1, 0.2]])
    monkeypatch.setattr(pipeline, "_SEMANTIC_FAILURE_UNTIL", 0.0, raising=False)

    assert pipeline._semantic("ทุนวิจัย", 2569) == (vector_hits, "alternate-embed:model")
