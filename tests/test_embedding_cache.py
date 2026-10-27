"""Regression tests for the optional on-disk semantic-vector cache.

The cache is only an accelerator.  A damaged or concurrently refreshed cache must
never make the deterministic/trigram answer path unusable or leave a half-written
JSON file behind.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.modules.knowledge import corpus as store


@pytest.fixture
def isolated_embedding_cache(monkeypatch, tmp_path):
    chunks = [
        {
            "id": "chunk-a",
            "text": "ทุนวิจัยสถาบัน",
            "status": "active",
            "verified": True,
            "fiscal_year": None,
        },
        {
            "id": "chunk-b",
            "text": "ทุนวิจัยในชั้นเรียน",
            "status": "active",
            "verified": True,
            "fiscal_year": None,
        },
    ]
    cache_path = tmp_path / "embeddings.json"
    monkeypatch.setattr(store, "CHUNKS", chunks)
    monkeypatch.setattr(store, "EMBEDDINGS", cache_path)
    monkeypatch.setattr(store, "EMBED_BATCH", len(chunks))
    monkeypatch.setattr(store.llm, "EMBED_MODEL", "test-embedding-model")
    monkeypatch.setattr(store, "_VECTORS", {})
    return cache_path, chunks


def _vectors_for(texts):
    return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]


def test_malformed_embedding_cache_is_rebuilt(isolated_embedding_cache, monkeypatch):
    cache_path, chunks = isolated_embedding_cache
    cache_path.write_text("{this is not json", encoding="utf-8")
    monkeypatch.setattr(store.llm, "embed", lambda texts, timeout: _vectors_for(texts))

    result = store.build_embeddings()

    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    assert result == {
        "model": "test-embedding-model",
        "chunks": len(chunks),
        "embedded_now": len(chunks),
    }
    assert set(saved["items"]) == {chunk["id"] for chunk in chunks}
    assert store.vectors_ready()


def test_embedding_response_count_must_match_batch(isolated_embedding_cache, monkeypatch):
    cache_path, _ = isolated_embedding_cache
    monkeypatch.setattr(store.llm, "embed", lambda texts, timeout: [[1.0, 2.0]])

    with pytest.raises(ValueError, match="2.*1|1.*2"):
        store.build_embeddings()

    assert not cache_path.exists()
    assert store._VECTORS == {}


def test_failed_atomic_replace_preserves_previous_cache(isolated_embedding_cache, monkeypatch):
    cache_path, chunks = isolated_embedding_cache
    previous = {
        "model": "test-embedding-model",
        "items": {
            chunks[0]["id"]: {
                "hash": store._text_hash(chunks[0]["text"]),
                "vec": [9.0, 9.0],
            }
        },
    }
    previous_text = json.dumps(previous)
    cache_path.write_text(previous_text, encoding="utf-8")
    monkeypatch.setattr(store.llm, "embed", lambda texts, timeout: _vectors_for(texts))

    def fail_replace(source, destination):
        raise OSError("simulated interrupted swap")

    monkeypatch.setattr(store.os, "replace", fail_replace)

    with pytest.raises(OSError, match="interrupted swap"):
        store.build_embeddings()

    assert cache_path.read_text(encoding="utf-8") == previous_text
    assert list(cache_path.parent.glob(f".{cache_path.name}.*.tmp")) == []


def test_concurrent_embedding_builds_share_one_refresh(isolated_embedding_cache, monkeypatch):
    _, chunks = isolated_embedding_cache
    rendezvous = threading.Barrier(2)
    calls = 0
    calls_lock = threading.Lock()

    def embed(texts, timeout):
        nonlocal calls
        with calls_lock:
            calls += 1
        try:
            rendezvous.wait(timeout=0.25)
        except threading.BrokenBarrierError:
            pass
        return _vectors_for(texts)

    monkeypatch.setattr(store.llm, "embed", embed)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: store.build_embeddings(), range(2)))

    assert calls == 1
    assert results[0]["embedded_now"] == len(chunks)
    assert results[1]["embedded_now"] == 0
    assert store.vectors_ready()


def test_mixed_dimension_cache_is_rebuilt_instead_of_silently_truncated(
    isolated_embedding_cache, monkeypatch
):
    cache_path, chunks = isolated_embedding_cache
    cache_path.write_text(
        json.dumps(
            {
                "model": "test-embedding-model",
                "items": {
                    chunks[0]["id"]: {
                        "hash": store._text_hash(chunks[0]["text"]),
                        "vec": [1.0, 2.0],
                    },
                    chunks[1]["id"]: {
                        "hash": store._text_hash(chunks[1]["text"]),
                        "vec": [3.0],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(store.llm, "embed", lambda texts, timeout: _vectors_for(texts))

    result = store.build_embeddings()

    assert result["embedded_now"] == len(chunks)
    assert {len(vector) for vector in store._VECTORS.values()} == {2}


def test_embedding_batch_rejects_inconsistent_vector_dimensions(
    isolated_embedding_cache, monkeypatch
):
    cache_path, _ = isolated_embedding_cache
    monkeypatch.setattr(
        store.llm,
        "embed",
        lambda texts, timeout: [[1.0, 2.0], [3.0]],
    )

    with pytest.raises(ValueError, match="dimension"):
        store.build_embeddings()

    assert not cache_path.exists()
    assert store._VECTORS == {}


def test_vector_search_rejects_a_query_with_the_wrong_dimension(
    isolated_embedding_cache, monkeypatch
):
    _, chunks = isolated_embedding_cache
    monkeypatch.setattr(
        store,
        "_VECTORS",
        {chunk["id"]: [1.0, 2.0] for chunk in chunks},
    )

    with pytest.raises(ValueError, match="dimension"):
        store.vector_search([1.0], None)


def test_stale_cache_entries_are_removed_even_when_no_embedding_is_missing(
    isolated_embedding_cache, monkeypatch
):
    cache_path, chunks = isolated_embedding_cache
    items = {
        chunk["id"]: {
            "hash": store._text_hash(chunk["text"]),
            "vec": [1.0, 2.0],
        }
        for chunk in chunks
    }
    items["removed-chunk"] = {"hash": "old", "vec": [3.0, 4.0]}
    cache_path.write_text(
        json.dumps({"model": "test-embedding-model", "items": items}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        store.llm,
        "embed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("up-to-date live chunks should not be re-embedded")
        ),
    )

    result = store.build_embeddings()

    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    assert result["embedded_now"] == 0
    assert set(saved["items"]) == {chunk["id"] for chunk in chunks}
