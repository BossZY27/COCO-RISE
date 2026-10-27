"""Architectural boundaries that must survive future maintenance."""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_app_root_contains_only_composition_files():
    assert {path.name for path in APP.glob("*.py")} == {"__init__.py", "bootstrap.py", "main.py"}


def test_v2_modules_do_not_import_legacy_http_modules():
    v2 = [path for path in (APP / "modules").rglob("*.py") if "legacy" not in path.parts]
    offenders = {
        str(path.relative_to(ROOT)): sorted(name for name in imports(path)
                                           if name.startswith("app.modules.legacy"))
        for path in v2
    }
    assert not {path: names for path, names in offenders.items() if names}


def test_removed_top_level_modules_are_not_imported():
    removed = {
        "store", "pipeline", "ingestion", "compat", "schemas", "api_admin",
        "answers", "answer_composer", "verification", "publication", "review",
        "llm", "audit", "limits", "identity", "chunking", "retrieval",
    }
    stale = {}
    for path in [*APP.rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]:
        found = sorted(name for name in imports(path)
                       if name.removeprefix("app.").split(".", 1)[0] in removed
                       and name.startswith("app."))
        if found:
            stale[str(path.relative_to(ROOT))] = found
    assert not stale
