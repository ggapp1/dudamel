from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest

from dudamel import App, Orchestrator, Runtime
from dudamel.config import Settings, TierConfig, WebConfig
from dudamel.llm.testing import Completion, FakeProvider, fake_text, fake_tool_call
from dudamel.web.api import create_api
from dudamel.web.auth import CSRF_HEADER
from dudamel.web.throttle import MAX_FAILURES

TOKEN = "s3cr3t-token"  # noqa: S105 — test fixture, not a real credential


def make_orc() -> Orchestrator:
    app = App("gym", description="d")

    @app.tool
    async def log_workout(exercise: str) -> str:
        """Record."""
        return f"ok {exercise}"

    @app.tool(confirm=True)
    async def wipe(reason: str) -> str:
        """Delete stuff."""
        return "wiped"

    @app.widget(title="Streak", renderer="markdown")
    async def streak() -> str:
        return "3 days"

    return Orchestrator(apps=[app])


def make_settings(tmp_path: Path, *, web: WebConfig | None = None) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/web.db",
        data_dir=tmp_path,
        llm_tiers={"standard": TierConfig(provider="fake", model="f")},
        web=web or WebConfig(),
    )


async def build(
    tmp_path: Path, script: list[Completion], *, web: WebConfig | None = None
) -> tuple[Runtime, httpx.ASGITransport]:
    orc = make_orc()
    settings = make_settings(tmp_path, web=web)
    rt = Runtime(orc, settings, providers={"standard": FakeProvider(script)})
    await rt.start()
    app = create_api(rt, settings)
    return rt, httpx.ASGITransport(app=app)


def client(transport: httpx.ASGITransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport, base_url="http://localhost")


@pytest.fixture
def token_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("DUDAMEL_WEB_TOKEN", TOKEN)
    return TOKEN


# --- auth: no token / wrong token -------------------------------------------


async def test_no_auth_returns_401(tmp_path: Path, token_env: str) -> None:
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        resp = await c.get("/api/widgets")
    assert resp.status_code == 401
    await rt.stop()


async def test_wrong_token_returns_401(tmp_path: Path, token_env: str) -> None:
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        resp = await c.get("/api/widgets", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401
    await rt.stop()


async def test_missing_configured_token_rejects_any_bearer(tmp_path: Path) -> None:
    """No DUDAMEL_WEB_TOKEN set at all: bearer auth can never succeed."""
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        resp = await c.get("/api/widgets", headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 401
    await rt.stop()


# --- bearer round-trip -------------------------------------------------------


async def test_bearer_chat_roundtrip(tmp_path: Path, token_env: str) -> None:
    rt, transport = await build(tmp_path, [fake_text("hello!")])
    async with client(transport) as c:
        resp = await c.post(
            "/api/chat",
            json={"text": "hi"},
            headers={"Authorization": f"Bearer {token_env}"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "hello!"
    assert body["pending_confirmation_id"] is None
    await rt.stop()


async def test_bearer_widgets_and_pending(tmp_path: Path, token_env: str) -> None:
    rt, transport = await build(
        tmp_path, [fake_tool_call("wipe", {"reason": "x"})], web=WebConfig()
    )
    headers = {"Authorization": f"Bearer {token_env}"}
    async with client(transport) as c:
        widgets = await c.get("/api/widgets", headers=headers)
        assert widgets.status_code == 200
        assert {w["id"] for w in widgets.json()} == {"streak"}

        chat = await c.post("/api/chat", json={"text": "wipe it"}, headers=headers)
        assert chat.status_code == 200
        confirmation_id = chat.json()["pending_confirmation_id"]
        assert confirmation_id

        pending = await c.get("/api/pending", headers=headers)
        assert pending.status_code == 200
        assert [p["id"] for p in pending.json()] == [confirmation_id]
        # This confirmation was raised by api_chat, which always resolves as
        # user_id="web" -- the same identity api_confirm resolves as, so
        # resolve_confirmation's requester check accepts it.
        assert pending.json()[0]["resolvable"] is True

        confirm = await c.post(
            f"/api/confirm/{confirmation_id}", json={"approved": False}, headers=headers
        )
        assert confirm.status_code == 200

        pending_after = await c.get("/api/pending", headers=headers)
        assert pending_after.json() == []
    await rt.stop()


async def test_pending_entries_report_channel_and_resolvability(
    tmp_path: Path, token_env: str
) -> None:
    """The list is deliberately unscoped -- reaching it already requires the
    web token, and the activity view shows every channel's tool calls to any
    authenticated operator. What the payload adds is the distinction between
    seeing an entry and being able to resolve it, so the dashboard stops
    offering actions that will always be rejected."""
    rt, transport = await build(tmp_path, [fake_tool_call("wipe", {"reason": "x"})])
    # Created directly through the runtime on a non-web channel/user, the way
    # a Telegram interface would -- api_chat itself only ever writes to
    # "web:*" channels, so this is otherwise unreachable from the HTTP API.
    reply = await rt.chat("telegram:1", "wipe it", user_id="someone-else")
    assert reply.pending_confirmation_id
    headers = {"Authorization": f"Bearer {token_env}"}
    async with client(transport) as c:
        resp = await c.get("/api/pending", headers=headers)
    assert resp.status_code == 200
    entries = resp.json()
    assert entries
    entry = entries[0]
    assert entry["channel"] == "telegram:1"
    # api_confirm always resolves as user_id="web"; resolve_confirmation
    # rejects any resolve where the resolver's user_id doesn't match the
    # confirmation's -- so a confirmation raised by "someone-else" is not
    # resolvable by this caller.
    assert entry["resolvable"] is False
    await rt.stop()


# --- session cookie login + CSRF ---------------------------------------------


async def test_login_sets_httponly_samesite_strict_cookie(tmp_path: Path, token_env: str) -> None:
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        resp = await c.post("/login", json={"token": token_env})
    assert resp.status_code == 200
    set_cookie = resp.headers.get("set-cookie", "")
    assert "HttpOnly" in set_cookie
    assert "samesite=strict" in set_cookie.lower()
    assert "csrf_token" in resp.json()
    await rt.stop()


async def test_login_wrong_token_401(tmp_path: Path, token_env: str) -> None:
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        resp = await c.post("/login", json={"token": "wrong"})
    assert resp.status_code == 401
    await rt.stop()


async def test_cookie_post_without_csrf_403_with_csrf_200(tmp_path: Path, token_env: str) -> None:
    rt, transport = await build(tmp_path, [fake_text("hello!")])
    async with client(transport) as c:
        login = await c.post("/login", json={"token": token_env})
        csrf_token = login.json()["csrf_token"]

        no_csrf = await c.post("/api/chat", json={"text": "hi"})
        assert no_csrf.status_code == 403

        with_csrf = await c.post(
            "/api/chat", json={"text": "hi"}, headers={CSRF_HEADER: csrf_token}
        )
        assert with_csrf.status_code == 200
        assert with_csrf.json()["text"] == "hello!"
    await rt.stop()


async def test_cookie_get_does_not_require_csrf(tmp_path: Path, token_env: str) -> None:
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        await c.post("/login", json={"token": token_env})
        resp = await c.get("/api/widgets")
    assert resp.status_code == 200
    await rt.stop()


# --- cookie Secure / __Host- / Max-Age ----------------------------------------


async def test_cookie_is_insecure_and_plainly_named_on_loopback(
    tmp_path: Path, token_env: str
) -> None:
    rt, transport = await build(tmp_path, [], web=WebConfig())  # default host: loopback
    async with client(transport) as c:
        resp = await c.post("/login", json={"token": token_env})
    cookie = resp.headers["set-cookie"]
    assert "dudamel_session=" in cookie
    assert "__Host-" not in cookie
    assert "Secure" not in cookie
    assert "Max-Age=86400" in cookie
    await rt.stop()


async def test_cookie_is_secure_and_host_prefixed_off_box(tmp_path: Path, token_env: str) -> None:
    """Auto-derives from the bind host: a static insecure default would be
    silent in exactly the deployment that needs it."""
    rt, transport = await build(tmp_path, [], web=WebConfig(host="0.0.0.0"))
    async with client(transport) as c:
        resp = await c.post("/login", json={"token": token_env})
    cookie = resp.headers["set-cookie"]
    assert "__Host-dudamel_session=" in cookie
    assert "Secure" in cookie
    assert "Path=/" in cookie
    assert "Domain=" not in cookie
    await rt.stop()


async def test_explicit_cookie_secure_overrides_the_host_derivation(
    tmp_path: Path, token_env: str
) -> None:
    rt, transport = await build(tmp_path, [], web=WebConfig(cookie_secure=True))
    async with client(transport) as c:
        resp = await c.post("/login", json={"token": token_env})
    assert "Secure" in resp.headers["set-cookie"]
    await rt.stop()


async def test_a_secure_session_cookie_still_authenticates(tmp_path: Path, token_env: str) -> None:
    """The rename must not break the round trip -- the authenticator has to
    read whichever name was issued. httpx's cookie jar won't replay a Secure
    cookie over the http:// test transport, so the cookie header is passed
    through explicitly here to exercise the actual read side."""
    rt, transport = await build(tmp_path, [fake_text("hi")], web=WebConfig(host="0.0.0.0"))
    async with client(transport) as c:
        login = await c.post("/login", json={"token": token_env})
        csrf = login.json()["csrf_token"]
        cookie_header = login.headers["set-cookie"].split(";")[0]

        pending = await c.get("/api/pending", headers={"Cookie": cookie_header})
        assert pending.status_code == 200

        chat = await c.post(
            "/api/chat",
            json={"text": "hi"},
            headers={"Cookie": cookie_header, CSRF_HEADER: csrf},
        )
        assert chat.status_code == 200
    await rt.stop()


# --- login throttling ---------------------------------------------------------


async def test_correct_token_succeeds_even_while_throttled(tmp_path: Path, token_env: str) -> None:
    """The ordering that matters: validate first, throttle only on failure.
    Otherwise six bad requests every five minutes lock the operator out."""
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        for _ in range(MAX_FAILURES):
            resp = await c.post("/login", json={"token": "wrong"})
            assert resp.status_code == 401
        throttled = await c.post("/login", json={"token": "wrong"})
        assert throttled.status_code == 429

        resp = await c.post("/login", json={"token": token_env})
        assert resp.status_code == 200
        assert "csrf_token" in resp.json()
    await rt.stop()


async def test_repeated_failures_return_429_with_retry_after(
    tmp_path: Path, token_env: str
) -> None:
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        for _ in range(MAX_FAILURES):
            await c.post("/login", json={"token": "wrong"})
        resp = await c.post("/login", json={"token": "wrong"})
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) > 0
    await rt.stop()


async def test_failed_bearer_auth_feeds_the_same_counter(tmp_path: Path, token_env: str) -> None:
    """The same secret is accepted on every API route, so throttling only the
    login endpoint would leave an attacker free to switch endpoints."""
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        for _ in range(MAX_FAILURES):
            await c.get("/api/pending", headers={"Authorization": "Bearer wrong"})
        resp = await c.post("/login", json={"token": "wrong"})
    assert resp.status_code == 429
    await rt.stop()


async def test_successful_bearer_auth_clears_accumulated_failures(
    tmp_path: Path, token_env: str
) -> None:
    """A legitimate API client that occasionally sends a stale token must not
    accumulate toward the throttle forever -- its own successes clear the
    counter, same as a successful /login does."""
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        for _ in range(MAX_FAILURES - 1):
            await c.get("/api/pending", headers={"Authorization": "Bearer wrong"})
        good = await c.get("/api/pending", headers={"Authorization": f"Bearer {token_env}"})
        assert good.status_code == 200

        # Without the clear, the accumulated failures above plus these would
        # trip the throttle partway through; with the clear, none of these
        # should see anything but 401.
        for _ in range(MAX_FAILURES - 1):
            resp = await c.post("/login", json={"token": "wrong"})
            assert resp.status_code == 401
    await rt.stop()


# --- host allowlist -----------------------------------------------------------


async def test_bad_host_header_returns_400(tmp_path: Path, token_env: str) -> None:
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        resp = await c.get("/health", headers={"host": "evil.example.com"})
    assert resp.status_code == 400
    await rt.stop()


async def test_good_host_header_allowed(tmp_path: Path, token_env: str) -> None:
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        resp = await c.get("/health", headers={"host": "localhost"})
    assert resp.status_code == 200
    await rt.stop()


# --- /health -------------------------------------------------------------------


async def test_health_open_no_auth(tmp_path: Path) -> None:
    rt, transport = await build(tmp_path, [])
    async with client(transport) as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    await rt.stop()


async def test_health_returns_503_when_db_ping_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Body shape is unchanged either way -- only the status code moves, so
    infra that keys off the HTTP status (load balancers, orchestrators)
    sees the failure without every /health caller having to parse JSON."""
    rt, transport = await build(tmp_path, [])

    async def broken_ping() -> None:
        raise RuntimeError("db is down")

    monkeypatch.setattr(rt, "db_ping", broken_ping)
    async with client(transport) as c:
        resp = await c.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "error"
    assert body["db"] is False
    assert "version" in body
    await rt.stop()


# --- startup guard -------------------------------------------------------------


def test_startup_guard_raises_for_non_loopback_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DUDAMEL_WEB_TOKEN", raising=False)
    orc = make_orc()
    settings = make_settings(tmp_path, web=WebConfig(host="0.0.0.0"))
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([])})
    with pytest.raises(RuntimeError, match="token"):
        create_api(rt, settings)


def test_startup_guard_allows_non_loopback_with_token(tmp_path: Path, token_env: str) -> None:
    orc = make_orc()
    settings = make_settings(tmp_path, web=WebConfig(host="0.0.0.0"))
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([])})
    create_api(rt, settings)  # must not raise


def test_loopback_without_token_warns_dashboard_login_impossible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("DUDAMEL_WEB_TOKEN", raising=False)
    orc = make_orc()
    settings = make_settings(tmp_path)  # default WebConfig(): host is loopback
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([])})
    with caplog.at_level(logging.WARNING):
        create_api(rt, settings)
    assert any(
        "dashboard login impossible until DUDAMEL_WEB_TOKEN is set" in r.message
        for r in caplog.records
    )


def test_loopback_with_token_does_not_warn(
    tmp_path: Path, token_env: str, caplog: pytest.LogCaptureFixture
) -> None:
    orc = make_orc()
    settings = make_settings(tmp_path)
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([])})
    with caplog.at_level(logging.WARNING):
        create_api(rt, settings)
    assert not any("dashboard login impossible" in r.message for r in caplog.records)


# --- channel namespace ---------------------------------------------------------


async def test_chat_rejects_a_channel_outside_the_web_namespace(
    tmp_path: Path, token_env: str
) -> None:
    """The channel names the conversation the message joins, and handling a
    message auto-declines that conversation's pending confirmations. Left
    unrestricted, a web caller could decline a Telegram user's pending action
    and write into their history."""
    rt, transport = await build(tmp_path, [fake_text("hi")])
    try:
        async with client(transport) as c:
            headers = {"Authorization": f"Bearer {TOKEN}"}
            bad = await c.post(
                "/api/chat",
                json={"text": "decline that", "channel": "telegram:12345"},
                headers=headers,
            )
            assert bad.status_code == 400
            assert "web:" in bad.json()["detail"]

            ok = await c.post(
                "/api/chat", json={"text": "hi", "channel": "web:other"}, headers=headers
            )
            assert ok.status_code == 200
    finally:
        await rt.stop()


async def test_chat_default_channel_is_accepted(tmp_path: Path, token_env: str) -> None:
    rt, transport = await build(tmp_path, [fake_text("hi")])
    try:
        async with client(transport) as c:
            resp = await c.post(
                "/api/chat", json={"text": "hi"}, headers={"Authorization": f"Bearer {TOKEN}"}
            )
            assert resp.status_code == 200
            assert resp.json()["text"] == "hi"
    finally:
        await rt.stop()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
def test_loopback_hosts_are_recognised(host: str) -> None:
    from dudamel.web.api import is_loopback_host

    assert is_loopback_host(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "example.com", ""])
def test_non_loopback_hosts_are_recognised(host: str) -> None:
    from dudamel.web.api import is_loopback_host

    assert not is_loopback_host(host)


async def test_localhost_bind_without_token_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Binding to `localhost` is a loopback bind. Refusing to start without a
    token there treated a safe configuration as an unsafe one."""
    monkeypatch.delenv("DUDAMEL_WEB_TOKEN", raising=False)
    orc = make_orc()
    settings = make_settings(tmp_path, web=WebConfig(host="localhost"))
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([])})
    await rt.start()
    try:
        create_api(rt, settings)  # must not raise
    finally:
        await rt.stop()
