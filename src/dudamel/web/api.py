"""JSON API surface. Intentionally thin: every route does request parsing ->
a single `Runtime` call -> response serialization. Zero business logic and
zero LLM calls live here.
"""

from __future__ import annotations

import ipaddress
import logging
import secrets
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dudamel._version import __version__
from dudamel.config import Settings
from dudamel.runtime import Runtime
from dudamel.web.auth import (
    SESSION_TTL,
    Authenticator,
    AuthVia,
    SessionStore,
    resolve_token,
    session_cookie_name,
)
from dudamel.web.throttle import FailedAuthThrottle

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


def _utcnow() -> datetime:
    # Naive UTC, matching how `dudamel.router` stores and compares
    # PendingConfirmation.expires_at -- a tz-aware value here would raise on
    # comparison against that column instead of ever comparing false.
    return datetime.now(UTC).replace(tzinfo=None)


def _client_key(request: Request) -> str:
    """The address the failed-auth throttle keys on.

    uvicorn's ``proxy_headers`` middleware (wired in ``dudamel.serve`` from
    ``settings.web.trusted_proxies``) already rewrites `request.client.host`
    to the forwarded address when the immediate peer is a trusted proxy, and
    leaves it as the peer's own address otherwise. Re-parsing
    `X-Forwarded-For` here would duplicate that trust decision -- and could
    disagree with it -- so this reads the address ASGI handed the app.
    """
    client = request.client
    # `request.client` is Optional per the ASGI spec, but uvicorn's real HTTP
    # protocol always populates it -- this fallback merges any such request
    # into one shared bucket, which is intentionally unreachable in practice
    # rather than a latent shared-throttle risk.
    return client.host if client is not None else "unknown"


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

    # Resolved once, and shared by /login's cookie name/flags and the
    # Authenticator's read side, so they can never disagree about which
    # cookie name is in play.
    cookie_secure = (
        settings.web.cookie_secure
        if settings.web.cookie_secure is not None
        else not is_loopback_host(settings.web.host)
    )

    sessions = SessionStore()
    throttle = FailedAuthThrottle()
    authenticate = Authenticator(settings, sessions, throttle, _client_key, cookie_secure)

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
    async def login(payload: LoginRequest, request: Request, response: Response) -> dict[str, str]:
        key = _client_key(request)
        expected = resolve_token(settings)
        # Validate BEFORE throttling, always. A correct token succeeds even
        # while throttled and clears the counter -- otherwise anyone who can
        # reach this endpoint could lock the operator out of their own
        # dashboard with a handful of wrong guesses.
        if expected is not None and secrets.compare_digest(
            payload.token.encode(), expected.encode()
        ):
            throttle.clear(key)
            session_id, csrf_token = sessions.create()
            response.set_cookie(
                session_cookie_name(cookie_secure),
                session_id,
                httponly=True,
                samesite="strict",
                secure=cookie_secure,
                path="/",
                max_age=int(SESSION_TTL.total_seconds()),
            )
            return {"csrf_token": csrf_token}
        if throttle.is_throttled(key):
            raise HTTPException(
                status_code=429,
                detail="too many failed attempts",
                headers={"Retry-After": str(throttle.retry_after(key))},
            )
        throttle.record_failure(key)
        raise HTTPException(status_code=401, detail="invalid token")

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
        """Every pending confirmation, across channels — deliberately unscoped.

        Reaching this endpoint already requires the web token, cookie auth is
        SameSite=Strict and safe-method-only, and a cross-user resolve is
        rejected by `resolve_confirmation` itself (it compares the resolving
        caller's user_id against the confirmation's own, inside the
        conversation lock). The activity view already exposes every
        channel's tool calls and arguments to any authenticated operator, so
        scoping this list would be inconsistent rather than safer.

        `resolvable` mirrors both grounds on which `resolve_confirmation`
        would actually apply this API's decision, for this API surface,
        which always resolves as user_id="web" (see api_confirm above):
        requester identity (the confirmation's own user_id must be "web")
        and TTL (its expires_at must not have passed -- past it,
        resolve_confirmation flips the row to "expired" and reports nothing
        was done instead of applying approve/decline, even for a matching
        user_id). It marks the entries this caller can actually act on right
        now, so the dashboard can show the rest as read-only instead of
        offering a button that will always be refused or silently no-op.
        The third ground resolve_confirmation checks, status != "pending",
        never applies here: this list is already filtered to status ==
        "pending".
        """
        now = _utcnow()
        entries = await runtime.list_pending_confirmations(channel)
        return [
            {
                "id": e["id"],
                "tool": e["tool"],
                "args": e["args"],
                "created_at": e["created_at"],
                "expires_at": e["expires_at"],
                "channel": e["channel"],
                "resolvable": e["user_id"] == "web" and e["expires_at"] >= now,
            }
            for e in entries
        ]

    return app
