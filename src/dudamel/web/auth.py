"""Bearer-token / session-cookie auth + CSRF enforcement for the web API.

A request is authenticated if it carries EITHER:
  - a valid ``Authorization: Bearer <token>`` header (constant-time compared
    via `secrets.compare_digest`), or
  - a valid session cookie (issued by ``POST /login``).

CSRF protection applies ONLY to the cookie path on state-changing (non-safe)
methods: a bearer-authed request already proves possession of a secret a
browser can't attach on its own, so the CSRF defense (which exists to stop a
browser from silently replaying cookies cross-site) doesn't apply to it.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import HTTPException, Request

from dudamel.config import Settings

SESSION_COOKIE = "dudamel_session"
CSRF_HEADER = "x-csrf-token"

_SESSION_TTL = timedelta(hours=24)
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

AuthVia = Literal["bearer", "session"]


def _now() -> datetime:
    return datetime.now(UTC)


def resolve_token(settings: Settings) -> str | None:
    """The configured bearer/login token, or None if its env var is unset/empty."""
    return os.environ.get(settings.web.token_env) or None


@dataclass
class _Session:
    csrf_token: str
    created_at: datetime


class SessionStore:
    """In-memory session table. dudamel always runs as one single process
    (see dudamel.serve.serve), which makes this sufficient — no cross-process
    session sharing is required, and sessions are intentionally lost on
    restart."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}

    def create(self) -> tuple[str, str]:
        """Issue a new session; returns (session_id, csrf_token)."""
        session_id = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(32)
        self._sessions[session_id] = _Session(csrf_token=csrf_token, created_at=_now())
        return session_id, csrf_token

    def csrf_for(self, session_id: str) -> str | None:
        """The session's CSRF token, or None if the session is missing/expired."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if _now() - session.created_at > _SESSION_TTL:
            del self._sessions[session_id]
            return None
        return session.csrf_token


class Authenticator:
    """FastAPI dependency: authenticates a request (bearer OR session
    cookie), enforcing CSRF for cookie-authed state-changing methods. Returns
    which mechanism authenticated the request; raises 401/403 otherwise."""

    def __init__(self, settings: Settings, sessions: SessionStore) -> None:
        self._settings = settings
        self._sessions = sessions

    async def __call__(self, request: Request) -> AuthVia:
        token = resolve_token(self._settings)
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            provided = auth_header.removeprefix("Bearer ")
            # compare_digest requires ASCII-only str (or bytes); encoding
            # sidesteps a TypeError on a non-ASCII header degrading auth to a
            # 500 instead of the intended 401.
            if token is not None and secrets.compare_digest(provided.encode(), token.encode()):
                return "bearer"
            raise HTTPException(status_code=401, detail="invalid bearer token")

        session_id = request.cookies.get(SESSION_COOKIE)
        if session_id is not None:
            csrf_token = self._sessions.csrf_for(session_id)
            if csrf_token is not None:
                if request.method not in _SAFE_METHODS:
                    provided_csrf = request.headers.get(CSRF_HEADER)
                    if provided_csrf is None or not secrets.compare_digest(
                        provided_csrf.encode(), csrf_token.encode()
                    ):
                        raise HTTPException(status_code=403, detail="missing or invalid CSRF token")
                return "session"

        raise HTTPException(status_code=401, detail="authentication required")
