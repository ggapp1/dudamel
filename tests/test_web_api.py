from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from dudamel import App, Orchestrator, Runtime
from dudamel.config import Settings, TierConfig, WebConfig
from dudamel.llm.testing import Completion, FakeProvider, fake_text, fake_tool_call
from dudamel.web.api import create_api
from dudamel.web.auth import CSRF_HEADER

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

        confirm = await c.post(
            f"/api/confirm/{confirmation_id}", json={"approved": False}, headers=headers
        )
        assert confirm.status_code == 200

        pending_after = await c.get("/api/pending", headers=headers)
        assert pending_after.json() == []
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
