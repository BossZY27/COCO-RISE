"""Documentation endpoints must describe the demo and MVP design separately."""
import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


ROOT = Path(__file__).resolve().parents[1]


def _refs(value):
    if isinstance(value, dict):
        ref = value.get("$ref")
        if ref:
            yield ref
        for item in value.values():
            yield from _refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _refs(item)


def _resolve(document, ref):
    assert ref.startswith("#/"), ref
    value = document
    for component in ref[2:].split("/"):
        component = component.replace("~1", "/").replace("~0", "~")
        value = value[component]
    return value


def test_demo_swagger_is_self_hosted_and_documents_admin_auth():
    with TestClient(app) as client:
        page = client.get("/docs")
        schema = client.get("/openapi.json")
        assets = client.get("/_swagger/swagger-ui-bundle.js")
        styles = client.get("/_swagger/swagger-ui.css")
        favicon = client.get("/_swagger/favicon-32x32.png")

    assert page.status_code == 200
    assert "/_swagger/swagger-ui-bundle.js" in page.text
    assert "/_swagger/favicon-32x32.png" in page.text
    assert "https://" not in page.text
    assert "script-src 'self' 'unsafe-inline'" in page.headers["content-security-policy"]
    assert schema.status_code == 200
    assert assets.status_code == 200
    assert styles.status_code == 200
    assert favicon.status_code == 200
    assert "application/json" in assets.headers["content-type"] or "javascript" in assets.headers["content-type"]
    spec = schema.json()
    assert spec["openapi"] == "3.0.3"
    assert spec["info"]["title"] == "RISE API"
    assert spec["components"]["securitySchemes"]["AdminToken"]["name"] == "X-Admin-Token"
    assert spec["paths"]["/v1/query"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/Query")
    assert "example" in spec["paths"]["/v1/query"]["post"]["responses"]["200"]["content"]["application/json"]
    assert "text/event-stream" in spec["paths"]["/v1/query/stream"]["post"]["responses"]["200"]["content"]
    assert "application/pdf" in spec["paths"]["/v1/admin/documents"]["post"]["requestBody"]["content"]
    assert spec["paths"]["/v1/admin/documents"]["post"]["security"] == [{"AdminToken": []}]


def test_design_swagger_is_read_only_and_every_reference_resolves():
    with TestClient(app) as client:
        page = client.get("/design/docs")
        response = client.get("/design/openapi.json")

    assert page.status_code == 200
    assert "supportedSubmitMethods" in page.text
    assert response.status_code == 200
    spec = response.json()
    assert spec["openapi"] == "3.0.3"
    assert "implemented" in spec["info"]["description"]
    for ref in _refs(spec):
        _resolve(spec, ref)
    assert spec["paths"]["/v2/documents"]["post"]["requestBody"]["content"]["application/pdf"]
    assert spec["paths"]["/v2/query"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]


def test_design_openapi_file_is_the_served_contract():
    saved = json.loads((ROOT / "docs" / "openapi-mvp.json").read_text(encoding="utf-8"))
    with TestClient(app) as client:
        live = client.get("/design/openapi.json").json()
    assert live == saved


def test_all_documentation_pages_and_specs_are_disabled_with_setting():
    env = os.environ.copy()
    env["RISE_EXPOSE_DOCS"] = "off"
    result = subprocess.run(
        [sys.executable, "-c", (
            "from fastapi.testclient import TestClient; from app.main import app; "
            "c=TestClient(app); print(','.join(str(c.get(p).status_code) for p in "
            "['/docs','/openapi.json','/design/docs','/design/openapi.json','/redoc']))"
        )],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().endswith("404,404,404,404,404")


def test_saved_contract_matches_current_application_schemas():
    from scripts.export_openapi import design_schema
    saved = json.loads((ROOT / "docs/openapi-mvp.json").read_text(encoding="utf-8"))
    assert saved == design_schema()
