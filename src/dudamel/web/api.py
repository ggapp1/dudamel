"""JSON API surface. Intentionally thin: every route does request parsing ->
a single `Runtime` call -> response serialization. Zero business logic and
zero LLM calls live here.
"""

from __future__ import annotations

import ipaddress
import logging
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

logger = logging.getLogger("dudamel.web.api")

__all__ = ["create_api", "is_loopback_host"]

# Every channel this surface may write to. The channel selects the
# conversation a message joins, and handling a message auto-declines that
# conversation's pending confirmations -- so an unconstrained channel would
# let a web caller reach into a Telegram user's conversation, decline the
# action they were asked to approve, and append to their history. The
# dashboard itself only ever uses "web:default".
WEB_CHANNEL_PREFIX = "web:"


class LoginRequest(BaseModel):
    token: str


class ChatRequest(BaseModel):
    text: str
    channel: str = "web:default"
    client_msg_id: str | None = None


class ConfirmRequest(BaseModel):
    approved: bool


def is_loopback_host(host: str) -> bool:
    """Whether binding to `host` keeps the surface on this machine.

    A literal comparison against "127.0.0.1" gets both directions wrong:
    "localhost" and "::1" are loopback binds that would be treated as
    off-box exposure, while the check that matters -- catching "0.0.0.0" and
    real interface addresses -- is unaffected by doing this properly.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # A hostname that isn't an address literal: not provably loopback,
        # so treat it as exposed. Refusing to start without a token is the
        # safe direction to be wrong in.
        return False


def create_api(runtime: Runtime, settings: Settings) -> FastAPI:
    """Build the FastAPI app. Raises RuntimeError at construction (never at
    request time) when the configured host is non-loopback and no web token
    is configured — refusing to expose an unauthenticated surface off-box."""
    token = resolve_token(settings)
    if not is_loopback_host(settings.web.host) and token is None:
        raise RuntimeError(
            f"refusing to start: binding to non-loopback host {settings.web.host!r} "
            f"requires a web token — set the {settings.web.token_env} environment variable"
        )
    if is_loopback_host(settings.web.host) and token is None:
        # Loopback-only is safe to start unauthenticated (the RuntimeError
        # above is what actually gates off-box exposure), but a dev who
        # forgets to set a token can't log into the dashboard at all — that's
        # easy to miss silently, so flag it loudly at startup instead.
        logger.warning("dashboard login impossible until %s is set", settings.web.token_env)

    sessions = SessionStore()
    authenticate = Authenticator(settings, sessions)

    app = FastAPI(title="dudamel", version=__version__)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.web.allowed_hosts)
    # Stashed so `dudamel.web.ui.add_ui()` can share this exact SessionStore:
    # a session issued by POST /login below must be recognized by both the
    # JSON API and the HTML dashboard pages.
    app.state.sessions = sessions

    @app.get("/health")
    async def health(response: Response) -> dict[str, Any]:
        try:
            await runtime.db_ping()
            db_ok = True
        except Exception:
            db_ok = False
        if not db_ok:
            # Body is intentionally unchanged either way -- only the status
            # code communicates liveness to infra (load balancers, container
            # orchestrators) that key off it rather than parsing the body.
            response.status_code = 503
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
        if not payload.channel.startswith(WEB_CHANNEL_PREFIX):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"channel {payload.channel!r} is outside this API's namespace; "
                    f"it must start with {WEB_CHANNEL_PREFIX!r}"
                ),
            )
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
