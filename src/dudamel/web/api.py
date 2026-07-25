"""JSON API surface (Plan 3 Task 3). THIN by Global Constraints: every route
does request parsing -> a single `Runtime` call -> response serialization.
Zero business logic and zero LLM calls live here.
"""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Response
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dudamel._version import __version__
from dudamel.config import Settings
from dudamel.runtime import Runtime
from dudamel.web.auth import (
    SESSION_COOKIE,
    Authenticator,
    AuthVia,
    SessionStore,
    resolve_token,
)

__all__ = ["create_api"]


class LoginRequest(BaseModel):
    token: str


class ChatRequest(BaseModel):
    text: str
    channel: str = "web:default"
    client_msg_id: str | None = None


class ConfirmRequest(BaseModel):
    approved: bool


def create_api(runtime: Runtime, settings: Settings) -> FastAPI:
    """Build the FastAPI app. Raises RuntimeError at construction (never at
    request time) when the configured host is non-loopback and no web token
    is configured — refusing to expose an unauthenticated surface off-box."""
    token = resolve_token(settings)
    if settings.web.host != "127.0.0.1" and token is None:
        raise RuntimeError(
            f"refusing to start: binding to non-loopback host {settings.web.host!r} "
            f"requires a web token — set the {settings.web.token_env} environment variable"
        )

    sessions = SessionStore()
    authenticate = Authenticator(settings, sessions)

    app = FastAPI(title="dudamel", version=__version__)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.web.allowed_hosts)
    # Stashed so `dudamel.web.ui.add_ui()` (Plan 3 Task 4) can share this
    # exact SessionStore: a session issued by POST /login below must be
    # recognized by both the JSON API and the HTML dashboard pages.
    app.state.sessions = sessions

    @app.get("/health")
    async def health() -> dict[str, Any]:
        try:
            await runtime.db_ping()
            db_ok = True
        except Exception:
            db_ok = False
        return {"status": "ok" if db_ok else "error", "version": __version__, "db": db_ok}

    @app.post("/login")
    async def login(payload: LoginRequest, response: Response) -> dict[str, str]:
        expected = resolve_token(settings)
        if expected is None or not secrets.compare_digest(
            payload.token.encode(), expected.encode()
        ):
            raise HTTPException(status_code=401, detail="invalid token")
        session_id, csrf_token = sessions.create()
        response.set_cookie(
            SESSION_COOKIE,
            session_id,
            httponly=True,
            samesite="strict",
        )
        return {"csrf_token": csrf_token}

    @app.post("/api/chat")
    async def api_chat(
        payload: ChatRequest, auth: AuthVia = Depends(authenticate)
    ) -> dict[str, Any]:
        reply = await runtime.chat(
            payload.channel,
            payload.text,
            user_id="web",
            client_msg_id=payload.client_msg_id,
        )
        return {"text": reply.text, "pending_confirmation_id": reply.pending_confirmation_id}

    @app.post("/api/confirm/{confirmation_id}")
    async def api_confirm(
        confirmation_id: str,
        payload: ConfirmRequest,
        auth: AuthVia = Depends(authenticate),
    ) -> dict[str, Any]:
        reply = await runtime.resolve_confirmation(
            confirmation_id, approved=payload.approved, user_id="web"
        )
        return {"text": reply.text, "pending_confirmation_id": reply.pending_confirmation_id}

    @app.get("/api/widgets")
    async def api_widgets(auth: AuthVia = Depends(authenticate)) -> list[dict[str, Any]]:
        return await runtime.render_widgets()

    @app.get("/api/pending")
    async def api_pending(
        channel: str | None = None, auth: AuthVia = Depends(authenticate)
    ) -> list[dict[str, Any]]:
        return await runtime.list_pending_confirmations(channel)

    return app
