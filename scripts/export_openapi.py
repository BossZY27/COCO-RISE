"""Generate the read-only v2 contract from the actual application schemas."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app


def design_schema():
    full = app.openapi()
    return {**full, "info": {**full["info"], "title": "RISE v2 API contract",
                              "description": "Read-only contract generated from the implemented /v2 routes."},
            "paths": {p: v for p, v in full["paths"].items() if p.startswith("/v2/")}}


if __name__ == "__main__":
    Path("docs/openapi-mvp.json").write_text(json.dumps(design_schema(), ensure_ascii=False, indent=2) + "\n")
