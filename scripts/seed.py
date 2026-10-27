import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.settings import Settings
from app.bootstrap import Services
from app.infrastructure.postgres.importer import import_legacy

if __name__ == "__main__":
    runtime = Services(Settings())
    try:
        print(import_legacy(runtime.database, runtime.files, Path("data"), runtime.settings.source_dir))
    finally:
        runtime.close()
