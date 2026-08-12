"""Server-rendered HTMX dashboard.

Interfaces here are intentionally thin: every page route does a session
check -> one or more `Runtime` read calls -> template render. Zero business
logic, zero LLM calls, and (this is the important bit for this module) zero
state changes live here.

Mounting: call ``add_ui(app, runtime, settings)`` AFTER ``create_api(runtime,
settings)`` has built ``app`` — it reads the ``SessionStore`` that
``create_api()`` stashes on ``app.state.sessions`` so a single ``POST
/login`` (the JSON API's own login route) issues a session cookie that both
the JSON API and these HTML pages recognize. ``add_ui`` never creates its
own session store or duplicates login logic; sessions stay strictly
``web/api.py``'s concern.

State changes (sending a chat message, approving/declining a pending
confirmation) are NOT re-implemented here. Each page ships a few lines of
vanilla JS that call the JSON API's existing, already-CSRF-protected
``/api/chat`` and ``/api/confirm/{id}`` endpoints directly from the browser,
using the session cookie already on the page and a CSRF token embedded in a
hidden form field (``#csrf-token``) — the standard "CSRF token embedded in
forms for cookie POSTs" mitigation for cookie-authenticated state changes.
ui.py's own routes are therefore all plain ``GET``s; it never proxies a POST
through Python.

HTMX is used only for its core strength — polling GETs — via
``hx-trigger="every Ns"`` + ``hx-select`` (dashboard: 30s, chat: 5s), so
periodic refresh never discards in-progress input (the chat textbox lives
outside the polled fragment).

Markdown widgets are rendered as pre-escaped plain text (Jinja's
``autoescape`` is on for all ``.html`` templates here) inside a styled
``<pre>`` block — there is no markdown parser in this module by design:
pulling a markdown library was explicitly out of scope for v1; "plain text
in a styled block" is the documented v1 behavior. Because autoescape
applies uniformly to every renderer's template output (stat/table/list/
markdown and the error card), this is also what makes a `<script>` in ANY
widget payload inert.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dudamel.config import Settings
from dudamel.runtime import Runtime
from dudamel.web.api import resolve_cookie_secure
from dudamel.web.auth import SessionStore, session_cookie_name

__all__ = ["add_ui"]

_WEB_DIR = Path(__file__).parent
_TEMPLATES = Jinja2Templates(directory=str(_WEB_DIR / "templates"))

_CHAT_CHANNEL = "web:default"


def add_ui(app: FastAPI, runtime: Runtime, settings: Settings) -> None:
    """Register the dashboard's HTML pages on `app` and mount its static
    assets (vendored htmx). Raises RuntimeError if `app` wasn't built by
    `create_api()` first — see the module docstring for why.

    `settings` is read to resolve the session cookie's name — the pages here
    must read back whichever cookie `create_api()`/`POST /login` issued, and
    that name derives from `resolve_cookie_secure(settings)` (see the
    `cookie_name` line below). There is no [web] dashboard knob of its own
    yet; the signature also stays symmetric with `create_api(runtime,
    settings)` for the single-process assembly (dudamel.serve.serve)."""
    sessions: SessionStore | None = getattr(app.state, "sessions", None)
    if sessions is None:
        raise RuntimeError(
            "add_ui(app, ...) requires create_api(runtime, settings) to have "
            "been called on `app` first — it shares create_api()'s SessionStore "
            "via app.state.sessions so a POST /login session is recognized here too."
        )
    app.mount("/static", StaticFiles(directory=str(_WEB_DIR / "static")), name="static")

    # The same resolution create_api() uses: whichever name /login issued is the
    # name these pages must read, or a secure deployment's dashboard would
    # loop back to /login forever despite a valid session cookie.
    cookie_name = session_cookie_name(resolve_cookie_secure(settings))

    def _session(request: Request) -> str | None:
        """The CSRF token for a valid session cookie on `request`, else None."""
        session_id = request.cookies.get(cookie_name)
        if session_id is None:
            return None
        return sessions.csrf_for(session_id)

    @app.get("/login")
    async def login_page(request: Request) -> Any:
        return _TEMPLATES.TemplateResponse(request, "login.html", {})

    @app.get("/")
    async def dashboard_page(request: Request) -> Any:
        csrf_token = _session(request)
        if csrf_token is None:
            return RedirectResponse("/login", status_code=303)
        widgets = await runtime.render_widgets()
        return _TEMPLATES.TemplateResponse(request, "dashboard.html", {"widgets": widgets})

    @app.get("/chat")
    async def chat_page(request: Request) -> Any:
        csrf_token = _session(request)
        if csrf_token is None:
            return RedirectResponse("/login", status_code=303)
        messages = await runtime.recent_messages(_CHAT_CHANNEL)
        pending = await runtime.list_pending_confirmations(_CHAT_CHANNEL)
        return _TEMPLATES.TemplateResponse(
            request,
            "chat.html",
            {"messages": messages, "pending": pending, "csrf_token": csrf_token},
        )

    @app.get("/activity")
    async def activity_page(request: Request) -> Any:
        if _session(request) is None:
            return RedirectResponse("/login", status_code=303)
        rows = await runtime.list_activity(100)
        return _TEMPLATES.TemplateResponse(request, "activity.html", {"rows": rows})

    @app.get("/jobs")
    async def jobs_page(request: Request) -> Any:
        if _session(request) is None:
            return RedirectResponse("/login", status_code=303)
        runs = await runtime.list_job_runs(50)
        jobs = runtime.list_jobs()
        return _TEMPLATES.TemplateResponse(request, "jobs.html", {"runs": runs, "jobs": jobs})
