"""Acceptance tests for dudamel/web/ui.py: the server-rendered HTMX
dashboard."""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from dudamel import App, Orchestrator, Runtime
from dudamel.config import Settings, TierConfig, WebConfig
from dudamel.db import Database
from dudamel.llm.testing import FakeProvider, fake_text, fake_tool_call
from dudamel.models_core import JobRun
from dudamel.web.api import create_api
from dudamel.web.ui import add_ui

TOKEN = "s3cr3t-token"  # noqa: S105 — test fixture, not a real credential
XSS_PAYLOAD = "<script>alert('xss')</script>"


def make_orc() -> Orchestrator:
    app = App("gym", description="d")

    @app.tool
    async def log_workout(exercise: str) -> str:
        """Record a workout."""
        return f"ok {exercise}"

    @app.widget(title="Streak", renderer="stat")
    async def streak() -> dict:
        return {"label": "Streak", "value": 3, "unit": "days"}

    @app.widget(title="Boom Widget", renderer="stat")
    async def boom() -> dict:
        raise RuntimeError("widget blew up")

    @app.widget(title="Notes", renderer="markdown")
    async def notes() -> str:
        return XSS_PAYLOAD

    @app.job(interval_seconds=3600)
    async def nightly_report() -> None:
        pass

    return Orchestrator(apps=[app])


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/web.db",
        data_dir=tmp_path,
        llm_tiers={"standard": TierConfig(provider="fake", model="f")},
        web=WebConfig(),
    )


async def build(tmp_path: Path, script: list) -> tuple[Runtime, Settings, httpx.ASGITransport]:
    orc = make_orc()
    settings = make_settings(tmp_path)
    rt = Runtime(orc, settings, providers={"standard": FakeProvider(script)})
    await rt.start()
    app = create_api(rt, settings)
    add_ui(app, rt, settings)
    return rt, settings, httpx.ASGITransport(app=app)


def client(transport: httpx.ASGITransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=transport, base_url="http://localhost", follow_redirects=False
    )


@pytest.fixture
def token_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("DUDAMEL_WEB_TOKEN", TOKEN)
    return TOKEN


async def login(c: httpx.AsyncClient, token: str) -> None:
    resp = await c.post("/login", json={"token": token})
    assert resp.status_code == 200


# --- auth: redirect to /login -----------------------------------------------


@pytest.mark.parametrize("path", ["/", "/chat", "/activity", "/jobs"])
async def test_unauthed_page_redirects_to_login(tmp_path: Path, token_env: str, path: str) -> None:
    rt, _settings, transport = await build(tmp_path, [])
    async with client(transport) as c:
        resp = await c.get(path)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"
    await rt.stop()


async def test_login_page_is_public(tmp_path: Path, token_env: str) -> None:
    rt, _settings, transport = await build(tmp_path, [])
    async with client(transport) as c:
        resp = await c.get("/login")
    assert resp.status_code == 200
    assert "Log in" in resp.text
    await rt.stop()


# --- dashboard: widget grid, error card, XSS escaping -----------------------


async def test_dashboard_shows_widget_title_and_error_card(tmp_path: Path, token_env: str) -> None:
    rt, _settings, transport = await build(tmp_path, [])
    async with client(transport) as c:
        await login(c, token_env)
        resp = await c.get("/")
    assert resp.status_code == 200
    assert "Streak" in resp.text  # successful widget's title
    # raising widget degrades to an error card, not a 500
    assert 'class="card error"' in resp.text
    assert "Boom Widget" in resp.text
    assert "widget blew up" in resp.text
    await rt.stop()


async def test_dashboard_escapes_script_in_markdown_widget(tmp_path: Path, token_env: str) -> None:
    rt, _settings, transport = await build(tmp_path, [])
    async with client(transport) as c:
        await login(c, token_env)
        resp = await c.get("/")
    assert resp.status_code == 200
    assert XSS_PAYLOAD not in resp.text  # never rendered as a live tag
    assert "&lt;script&gt;" in resp.text  # escaped instead
    await rt.stop()


# --- chat: message text, working embedded CSRF token ------------------------


async def test_chat_page_shows_message_text(tmp_path: Path, token_env: str) -> None:
    rt, _settings, transport = await build(tmp_path, [fake_text("hello dashboard reply")])
    await rt.chat("web:default", "hello from the dashboard", user_id="web")
    async with client(transport) as c:
        await login(c, token_env)
        resp = await c.get("/chat")
    assert resp.status_code == 200
    assert "hello from the dashboard" in resp.text
    assert "hello dashboard reply" in resp.text
    await rt.stop()


async def test_chat_page_embedded_csrf_token_actually_works(tmp_path: Path, token_env: str) -> None:
    """The hidden #csrf-token field the chat page ships must be a genuine,
    currently-valid CSRF token for THIS session — usable against the real
    /api/chat CSRF check, not a decorative placeholder."""
    rt, _settings, transport = await build(tmp_path, [fake_text("ok!")])
    async with client(transport) as c:
        await login(c, token_env)
        chat_resp = await c.get("/chat")
        match = re.search(r'id="csrf-token" value="([^"]+)"', chat_resp.text)
        assert match, "chat page must embed a #csrf-token hidden field"
        csrf_token = match.group(1)

        no_csrf = await c.post("/api/chat", json={"text": "hi"})
        assert no_csrf.status_code == 403

        with_csrf = await c.post(
            "/api/chat", json={"text": "hi"}, headers={"x-csrf-token": csrf_token}
        )
        assert with_csrf.status_code == 200
    await rt.stop()


async def test_pending_confirmation_approve_deny_buttons_present(
    tmp_path: Path, token_env: str
) -> None:
    app = App("gym", description="d")

    @app.tool(confirm=True)
    async def wipe(reason: str) -> str:
        """Delete stuff."""
        return "wiped"

    @app.widget(title="Streak", renderer="stat")
    async def streak() -> dict:
        return {"label": "Streak", "value": 3}

    orc = Orchestrator(apps=[app])
    settings = make_settings(tmp_path)
    script = [fake_tool_call("wipe", {"reason": "x"})]
    rt = Runtime(orc, settings, providers={"standard": FakeProvider(script)})
    await rt.start()
    api_app = create_api(rt, settings)
    add_ui(api_app, rt, settings)
    transport = httpx.ASGITransport(app=api_app)

    await rt.chat("web:default", "wipe it", user_id="web")
    async with client(transport) as c:
        await login(c, TOKEN)
        resp = await c.get("/chat")
    assert resp.status_code == 200
    assert "wipe" in resp.text
    assert "resolveConfirm(" in resp.text
    await rt.stop()


# --- activity: tool name -----------------------------------------------------


async def test_activity_page_shows_tool_name(tmp_path: Path, token_env: str) -> None:
    rt, _settings, transport = await build(
        tmp_path, [fake_tool_call("log_workout", {"exercise": "bench"}), fake_text("logged!")]
    )
    await rt.chat("web:default", "log a bench workout", user_id="web")
    async with client(transport) as c:
        await login(c, token_env)
        resp = await c.get("/activity")
    assert resp.status_code == 200
    assert "log_workout" in resp.text
    assert "ok" in resp.text
    # The row carries the surface it came from, and this one came from the
    # router acting for the model -- rendered as its own cell.
    assert "<td>router</td>" in resp.text
    await rt.stop()


# --- jobs: registered job id + recent runs ----------------------------------


async def test_jobs_page_shows_registered_job_id(tmp_path: Path, token_env: str) -> None:
    rt, _settings, transport = await build(tmp_path, [])
    async with client(transport) as c:
        await login(c, token_env)
        resp = await c.get("/jobs")
    assert resp.status_code == 200
    assert "gym.nightly_report" in resp.text
    await rt.stop()


async def test_jobs_page_shows_recorded_run(tmp_path: Path, token_env: str) -> None:
    rt, settings, transport = await build(tmp_path, [])
    db = Database(settings.database_url)
    async with db.session() as s:
        s.add(JobRun(job_id="gym.nightly_report", status="ok", detail=None))
    async with client(transport) as c:
        await login(c, token_env)
        resp = await c.get("/jobs")
    assert resp.status_code == 200
    assert "gym.nightly_report" in resp.text
    assert "ok" in resp.text
    await db.dispose()
    await rt.stop()


# --- add_ui() wiring guard ----------------------------------------------------


def test_add_ui_without_create_api_raises(tmp_path: Path) -> None:
    orc = make_orc()
    settings = make_settings(tmp_path)
    rt = Runtime(orc, settings, providers={"standard": FakeProvider([])})
    with pytest.raises(RuntimeError, match="create_api"):
        add_ui(FastAPI(), rt, settings)
