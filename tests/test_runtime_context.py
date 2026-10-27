"""Compatibility aliases must use the same database evidence snapshot as v1."""
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.main import RuntimeContext
from app.modules.knowledge import corpus as store
from app.modules.knowledge.context import snapshot
from app.modules.query.engine import _current_corpus
from app.core.runtime import services


@pytest.mark.parametrize("path", ["/api/query", "/v1/query", "/v1/query/stream"])
def test_compat_query_pins_snapshot_and_still_rejects_changes(path):
    corpus = SimpleNamespace(revision=42)
    current = [42]
    runtime = SimpleNamespace(
        corpora=SimpleNamespace(load=lambda: corpus),
        repository=SimpleNamespace(corpus_revision=lambda: current[0]),
    )

    async def endpoint(request):
        assert services.get() is runtime
        assert store.view() is corpus
        return JSONResponse({"current": _current_corpus()})

    app = Starlette(routes=[Route(path, endpoint, methods=["POST"])])
    app.state.services = runtime
    with TestClient(RuntimeContext(app, app)) as client:
        assert client.post(path).json() == {"current": True}
        current[0] = 43
        assert client.post(path).json() == {"current": False}
    assert snapshot.get() is None
    assert services.get() is None
