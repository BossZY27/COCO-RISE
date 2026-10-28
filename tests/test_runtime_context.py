"""The public app exposes one API and no JSON demo or persona fallback."""
from fastapi.testclient import TestClient
from app.core.settings import Settings
import app.main as main


def test_only_canonical_paths_and_explicit_missing_database(monkeypatch):
    monkeypatch.setattr(main, "Settings", lambda: Settings(_env_file=None, database_url=""))
    with TestClient(main.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/ready").status_code == 503
        for path in ("/v1/query", "/v2/query", "/api/update", "/v1/query/stream"):
            assert client.post(path, json={"question":"มีทุนอะไรบ้าง"}).status_code == 404
        assert client.get("/v1/personas").status_code == 404
        assert client.get("/healthz").status_code == 404
        schema = client.get("/openapi.json").json()
        assert all(path.startswith("/api/") for path in schema["paths"])
        assert client.get("/admin").headers["cache-control"] == "no-store, private"
        assert "persona_id" not in schema["components"]["schemas"]["QueryInput"]["properties"]
