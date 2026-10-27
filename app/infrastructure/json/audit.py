"""Append one JSON line per answered question.

Two jobs. It is the record a fund officer needs when someone disputes an answer the system gave, and it
is the queue of real questions to add to data/eval.json so the test set grows from actual use.

It stores the question, the slots, the verdict and the citations. It does not store the profile fields
themselves, only the id that points at them, because grant history and employment type are personal data
under PDPA and they already live in the system of record.
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path

LOG = Path(os.getenv("RISE_AUDIT_LOG", Path(__file__).resolve().parents[3] / "logs" / "queries.jsonl"))
ENABLED = os.getenv("RISE_AUDIT", "on") != "off"
MAX_TAIL_ROWS = max(1, int(os.getenv("RISE_AUDIT_MAX_TAIL", "200")))
MAX_LOG_BYTES = max(1, int(os.getenv("RISE_AUDIT_MAX_BYTES", str(10 * 1024 * 1024))))
MAX_BACKUPS = max(0, int(os.getenv("RISE_AUDIT_BACKUPS", "1")))
MAX_TAIL_READ_BYTES = max(
    1024,
    int(os.getenv("RISE_AUDIT_TAIL_READ_BYTES", str(2 * 1024 * 1024))),
)
_LOCK = threading.Lock()


def _backup_path(index: int) -> Path:
    return LOG.with_name(f"{LOG.name}.{index}")


def _rotate_if_needed(incoming_bytes: int) -> None:
    """Keep the append-only audit trail bounded while retaining recent history."""
    if not LOG.exists() or LOG.stat().st_size + incoming_bytes <= MAX_LOG_BYTES:
        return
    if MAX_BACKUPS == 0:
        LOG.unlink()
        return
    for index in range(MAX_BACKUPS, 1, -1):
        source = _backup_path(index - 1)
        if source.exists():
            os.replace(source, _backup_path(index))
    os.replace(LOG, _backup_path(1))


def _read_bounded_suffix(path: Path) -> bytes:
    """Read at most the configured suffix and discard a split first line."""
    try:
        with path.open("rb") as handle:
            length = handle.seek(0, os.SEEK_END)
            start = max(0, length - MAX_TAIL_READ_BYTES)
            handle.seek(start)
            data = handle.read(MAX_TAIL_READ_BYTES)
    except OSError:
        return b""
    if start:
        separator = data.find(b"\n")
        return data[separator + 1:] if separator >= 0 else b""
    return data


def record(result: dict, user_id: str) -> None:
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.record_query(result, user_id)
    if not ENABLED:
        return
    from app.core import security as limits
    raw_slots = result.get("slots")
    safe_slots = dict(raw_slots) if isinstance(raw_slots, dict) else raw_slots
    if isinstance(safe_slots, dict) and isinstance(safe_slots.get("standalone_question"), str):
        safe_slots["standalone_question"] = limits.redact_pii(safe_slots["standalone_question"])
    row = {"at": datetime.now().isoformat(timespec="seconds"), "user_id": user_id,
           "question": limits.redact_pii(result.get("question")), "slots": safe_slots,
           "verdict": result.get("ruling", {}).get("verdict"), "answer": result.get("answer"),
           "citations": result.get("citations"), "answer_source": result.get("answer_source"),
           "blocked": result.get("verify", {}).get("ok") is False,
           "timings_ms": result.get("timings_ms")}
    line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with _LOCK:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            _rotate_if_needed(len(line))
            with LOG.open("ab") as handle:
                handle.write(line)
    except OSError:
        pass          # a full disk must not take the answer away from the person waiting for it


def tail(limit: int = 20) -> list[dict]:
    from app.core.runtime import services as runtime_services
    runtime = runtime_services.get()
    if runtime is not None:
        return runtime.legacy.recent(limit)
    size = max(0, min(int(limit), MAX_TAIL_ROWS))
    if size == 0:
        return []
    rows: list[dict] = []
    with _LOCK:
        paths = [LOG, *(_backup_path(index) for index in range(1, MAX_BACKUPS + 1))]
        for path in paths:
            data = _read_bounded_suffix(path)
            for raw_line in reversed(data.splitlines()):
                if not raw_line.strip():
                    continue
                try:
                    line = raw_line.decode("utf-8")
                    row = json.loads(line)
                except (UnicodeError, json.JSONDecodeError):
                    continue
                if isinstance(row, dict):
                    rows.append(row)
                if len(rows) >= size:
                    break
            if len(rows) >= size:
                break
    rows.reverse()
    return rows
