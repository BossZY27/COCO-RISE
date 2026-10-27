"""Run separately: python -m app.workers.main. Heartbeats use independent sessions."""
import logging
import signal
import threading
import uuid
from app.bootstrap import Services
from app.core.settings import Settings
from app.infrastructure.files.extraction import TesseractExtractor

log = logging.getLogger(__name__)


def process_one(services, owner, extractor=None):
    repo = services.repository
    job = repo.claim(owner)
    if job is None:
        return False
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(services.settings.job_lease_seconds / 3):
            try:
                if not repo.heartbeat(job["id"], owner):
                    return
            except Exception:
                log.exception("Job heartbeat failed")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        if job["kind"] == "index":
            services.vectors.build(services.ollama)
            repo.finish(job["id"], owner)
        else:
            pages = (extractor or TesseractExtractor()).extract(repo.job_source(job))
            repo.finish(job["id"], owner, pages=pages)
    except Exception as exc:
        log.exception("Job failed: %s", job["id"])
        repo.finish(job["id"], owner, error=f"{type(exc).__name__}: {exc}")
    finally:
        stopped.set()
        thread.join(timeout=2)
    return True


def main():
    logging.basicConfig(level=logging.INFO)
    services = Services(Settings())
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    owner = uuid.uuid4()
    try:
        while not stop.is_set():
            if not process_one(services, owner):
                stop.wait(1)
    finally:
        services.close()


if __name__ == "__main__":
    main()
