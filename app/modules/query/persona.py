"""Resolve the optional v1 demo persona before eligibility evaluation.

Eligibility depends on the asker, so every answer needs a profile. Right now the profiles are the five
fictional personas in data/personas.json. When the university directory is connected, replace
`lookup` with a call to it and leave the rest of the system alone: the pipeline only ever sees the
dict that comes back from here.

The profile fields the rules actually read are staff_type, department, and grants_this_year. A real
provider must return those three, and may return anything else it likes.
"""
import os

from app.modules.knowledge import corpus as store

SOURCE = os.getenv("RISE_PROFILE_SOURCE", "demo")      # demo reads personas.json
REQUIRED = ("staff_type", "department", "grants_this_year")


def lookup(user_id: str) -> dict | None:
    """Return the profile for this user, or None when the directory does not know them."""
    if SOURCE == "demo":
        return store.view().PERSONAS.get(user_id)
    raise NotImplementedError(
        f"RISE_PROFILE_SOURCE={SOURCE} ยังไม่มีตัวเชื่อม ให้เขียน lookup ที่ยิงไปยังฐานข้อมูลบุคลากรของมหาวิทยาลัย")


def resolve(header_user_id: str | None, body_persona_id: str | None) -> tuple[str, dict | None]:
    """Header wins over the request body, because a caller must not be able to ask as someone else.

    In production the header is set by the reverse proxy after it validates the session, never by the
    browser. The body field stays only so the demo page can switch personas.
    """
    user_id = header_user_id or body_persona_id or ""
    return user_id, lookup(user_id) if user_id else None


def missing_fields(profile: dict | None) -> list[str]:
    if not profile:
        return list(REQUIRED)
    return [f for f in REQUIRED if profile.get(f) in (None, "")]
