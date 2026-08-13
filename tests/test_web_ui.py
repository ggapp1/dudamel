"""Acceptance tests for dudamel/web/ui.py: the server-rendered HTMX
dashboard."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from dudamel import App, Orchestrator, Runtime
from dudamel.config import HomeConfig, HomeSection, Settings, TierConfig, WebConfig
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


# --- dashboard: embedded CSRF token for card actions -------------------------


async def test_dashboard_embedded_csrf_token_actually_works(tmp_path: Path, token_env: str) -> None:
    """The dashboard's card buttons POST to /api/action/{tool} from the
    browser under the session cookie, so the page has to ship a genuine,
    currently-valid CSRF token — the same hidden field the chat page uses."""
    rt, _settings, transport = await build(tmp_path, [])
    async with client(transport) as c:
        await login(c, token_env)
        resp = await c.get("/")
        assert resp.status_code == 200
        match = re.search(r'id="csrf-token" value="([^"]+)"', resp.text)
        assert match, "dashboard must embed a #csrf-token hidden field"

        no_csrf = await c.post("/api/action/log_workout", json={"args": {"exercise": "squat"}})
        assert no_csrf.status_code == 403

        # log_workout carries no action label, so the token gets it past the
        # CSRF gate and no further -- which is what proves the token is real.
        with_csrf = await c.post(
            "/api/action/log_workout",
            json={"args": {"exercise": "squat"}},
            headers={"x-csrf-token": match.group(1)},
        )
        assert with_csrf.status_code == 404
    await rt.stop()


# --- dashboard: action buttons and configured sections -----------------------

HOSTILE_ARG = "</script><img src=x onerror=alert(1)>"
# A right-to-left override in front of reversed text: a browser draws this as
# "Archive", on a button wired to a tool that deletes.
SPOOFED_LABEL = "\u202eegruP"


def make_tasks_orc() -> Orchestrator:
    """An app whose cards carry buttons: a per-item action, a confirming one,
    one whose argument value is hostile, and a card-level action."""
    app = App("tasks", description="d")

    @app.tool(action="Done")
    async def complete(id: int) -> str:
        """Complete a task."""
        return f"done {id}"

    @app.tool(confirm=True, action="Delete")
    async def wipe(id: int) -> str:
        """Delete a task."""
        return "gone"

    @app.tool(action="Note")
    async def note(text: str) -> str:
        """Attach a note to a task."""
        return "noted"

    @app.tool(action="Refresh")
    async def refresh() -> str:
        """Refresh the list."""
        return "refreshed"

    @app.tool(confirm=True, action=SPOOFED_LABEL)
    async def purge(id: int) -> str:
        """Delete everything."""
        return "purged"

    @app.widget(title="Today", renderer="list", actions=["refresh"])
    async def today() -> list[dict]:
        return [
            {"title": "Buy milk", "action": {"tool": "complete", "args": {"id": 1}}},
            {"title": "Old task", "action": {"tool": "wipe", "args": {"id": 2}}},
            {"title": "Feed item", "action": {"tool": "note", "args": {"text": HOSTILE_ARG}}},
            {"title": "Everything", "action": {"tool": "purge", "args": {"id": 3}}},
        ]

    @app.widget(title="Notes", renderer="markdown")
    async def scratch() -> str:
        return "some notes"

    return Orchestrator(apps=[app])


async def build_tasks(
    tmp_path: Path, home: HomeConfig | None = None
) -> tuple[Runtime, httpx.ASGITransport]:
    settings = make_settings(tmp_path)
    if home is not None:
        settings = settings.model_copy(update={"home": home})
    rt = Runtime(make_tasks_orc(), settings, providers={"standard": FakeProvider([])})
    await rt.start()
    api_app = create_api(rt, settings)
    add_ui(api_app, rt, settings)
    return rt, httpx.ASGITransport(app=api_app)


async def dashboard_body(tmp_path: Path, token: str, home: HomeConfig | None = None) -> str:
    rt, transport = await build_tasks(tmp_path, home)
    async with client(transport) as c:
        await login(c, token)
        resp = await c.get("/")
    assert resp.status_code == 200
    await rt.stop()
    return resp.text


class _ButtonCollector(HTMLParser):
    """Collects every <button>'s attributes with a real HTML parser, so an
    attribute value that terminates its own quoting is visible as the broken
    markup it is rather than passing a substring check."""

    def __init__(self) -> None:
        super().__init__()
        self.buttons: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "button":
            self.buttons.append(dict(attrs))


def action_buttons(body: str) -> dict[str, dict[str, str | None]]:
    collector = _ButtonCollector()
    collector.feed(body)
    return {b["data-tool"]: b for b in collector.buttons if b.get("data-tool")}


async def test_per_item_action_renders_a_button(tmp_path: Path, token_env: str) -> None:
    body = await dashboard_body(tmp_path, token_env)
    assert "<button" in body
    assert 'data-tool="complete"' in body
    button = action_buttons(body)["complete"]
    assert button["data-label"] == "Done"


async def test_button_arguments_survive_the_attribute_intact(
    tmp_path: Path, token_env: str
) -> None:
    """What the browser POSTs is `JSON.parse(button.dataset.args)`, so the
    attribute has to parse back to exactly the resolved arguments."""
    body = await dashboard_body(tmp_path, token_env)
    buttons = action_buttons(body)
    assert json.loads(buttons["complete"]["data-args"] or "") == {"id": 1}
    assert json.loads(buttons["note"]["data-args"] or "") == {"text": HOSTILE_ARG}
    assert json.loads(buttons["refresh"]["data-args"] or "") == {}


async def test_confirming_action_button_is_marked_and_a_plain_one_is_not(
    tmp_path: Path, token_env: str
) -> None:
    buttons = action_buttons(await dashboard_body(tmp_path, token_env))
    assert "data-confirm" in buttons["wipe"]
    assert "data-confirm" not in buttons["complete"]


async def test_card_level_action_renders_a_button(tmp_path: Path, token_env: str) -> None:
    body = await dashboard_body(tmp_path, token_env)
    assert 'data-tool="refresh"' in body
    assert 'class="card-actions"' in body


async def test_configured_sections_render_their_titles_in_order(
    tmp_path: Path, token_env: str
) -> None:
    home = HomeConfig(
        section=[
            HomeSection(title="Scratchpad", widgets=["tasks.scratch"]),
            HomeSection(title="Chores", widgets=["tasks.today"]),
        ]
    )
    body = await dashboard_body(tmp_path, token_env, home)
    assert "Scratchpad" in body
    assert "Chores" in body
    # configured order, not registration order (today registers before scratch)
    assert body.index("Scratchpad") < body.index("Chores")


async def test_a_bidi_override_never_reaches_the_page(tmp_path: Path, token_env: str) -> None:
    """Autoescape stops a label breaking out of its attribute and says nothing
    about which direction the characters inside it are drawn in. `data-label`
    is not decoration: it is the string `window.confirm` asks about and the
    string the aria-live region announces, so a label that reads "Archive" on a
    delete button would spoof the one mis-tap protection a confirming action
    has here. Asserted on the whole document, because the label is emitted
    twice -- as the attribute and as the button's own text."""
    body = await dashboard_body(tmp_path, token_env)
    assert "\u202e" not in body
    button = action_buttons(body)["purge"]
    assert button["data-label"] == "egruP"
    assert ">egruP</button>" in body


async def test_dashboard_ships_the_duplicate_mutation_guards(
    tmp_path: Path, token_env: str
) -> None:
    """A string check, not a behaviour check: the guards that keep a card's
    button from being swapped out from under a running action live in
    `dashboard.html`'s JavaScript, which this suite cannot execute. This
    asserts only that each load-bearing piece is still present in the page a
    logged-in user is served -- it proves nothing about what they do, and
    exists so deleting one is a red test rather than a silent regression."""
    body = await dashboard_body(tmp_path, token_env)
    assert "htmx:beforeSwap" in body
    assert "event.detail.shouldSwap = false" in body
    assert "htmx:beforeRequest" in body
    assert "event.preventDefault()" in body
    assert "button.disabled = true" in body
    assert 'id="action-status" role="status" aria-live="polite"' in body
    # Disabling a button is half of it: a failed action that never re-enables
    # its button, or never says what happened, leaves an operator with a dead
    # control and no reason for it.
    assert "button.disabled = false" in body
    assert "announce(label + ' failed: '" in body


async def test_hostile_action_argument_is_escaped_in_the_button_attribute(
    tmp_path: Path, token_env: str
) -> None:
    """An action's arguments can come from a feed or a fetched page, so a value
    carrying markup must not be able to close the attribute or the script."""
    body = await dashboard_body(tmp_path, token_env)
    assert 'data-tool="note"' in body  # the hostile value really did render
    assert "<img src=x" not in body
    assert "</script><img" not in body
