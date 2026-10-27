import hmac
from typing import Annotated
from fastapi import Depends, Header, HTTPException, Query, Request


def services(request: Request):
    runtime = getattr(request.app.state, "services", None)
    if runtime is None:
        raise HTTPException(503, "Configure RISE_DATABASE_URL and apply migrations")
    return runtime


def admin(request: Request, x_admin_token: str | None = Header(default=None)):
    expected = request.app.state.settings.admin_token
    if not expected or not x_admin_token or not hmac.compare_digest(expected, x_admin_token):
        raise HTTPException(401, "Valid X-Admin-Token required")


Admin = Annotated[None, Depends(admin)]
Limit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0)]
