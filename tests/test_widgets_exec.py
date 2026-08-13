"""Acceptance tests for dudamel/widgets.py::run_widget, including its
timeout enforcement."""

import asyncio

from dudamel.app import App
from dudamel.contract.types import Tool, Widget
from dudamel.widgets import run_widget


def make_widget(
    fn, *, renderer: str = "stat", id: str = "w1", title: str = "W", timeout: float = 15.0
) -> Widget:
    return Widget(id=id, app_name="app", title=title, renderer=renderer, fn=fn, timeout=timeout)


async def test_run_widget_stat_success() -> None:
    async def fn() -> dict:
        return {"label": "Vol", "value": 12, "unit": "kg"}

    out = await run_widget(make_widget(fn))
    assert out == {
        "id": "w1",
        "qualified_id": "app.w1",
        "title": "W",
        "renderer": "stat",
        "data": {"label": "Vol", "value": 12, "unit": "kg", "delta": None},
        "actions": [],
    }


async def test_run_widget_table_success() -> None:
    async def fn() -> dict:
        return {"columns": ["a"], "rows": [[1]]}

    out = await run_widget(make_widget(fn, renderer="table"))
    assert out["data"] == {"columns": ["a"], "rows": [[1]]}


async def test_run_widget_list_success_flattens_models_to_dicts() -> None:
    async def fn() -> list:
        return [{"title": "t", "url": "http://x"}]

    out = await run_widget(make_widget(fn, renderer="list"))
    # validate_widget_payload returns pydantic ListItem models; run_widget must
    # hand back plain JSON-safe data, not pydantic instances.
    assert out["data"] == [{"title": "t", "subtitle": None, "url": "http://x", "action": None}]


async def test_run_widget_markdown_success() -> None:
    async def fn() -> str:
        return "# hi"

    out = await run_widget(make_widget(fn, renderer="markdown"))
    assert out["data"] == "# hi"


async def test_run_widget_raising_widget_yields_error_shape() -> None:
    async def fn() -> dict:
        raise RuntimeError("boom")

    out = await run_widget(make_widget(fn))
    assert out["error"] == "boom"
    assert "data" not in out


async def test_run_widget_invalid_payload_yields_error_shape() -> None:
    async def fn() -> dict:
        return {"not": "a stat payload"}

    out = await run_widget(make_widget(fn))
    assert "error" in out
    assert "data" not in out


async def test_run_widget_timeout_yields_error_shape() -> None:
    async def fn() -> dict:
        await asyncio.sleep(5)
        return {"label": "x", "value": 1}

    out = await run_widget(make_widget(fn, timeout=0.05))
    assert out["error"] == "widget timed out after 0.05s"
    assert "data" not in out


async def test_run_widget_that_raises_its_own_timeouterror_reports_the_real_message() -> None:
    """A widget whose fn() raises TimeoutError itself (e.g. an OS-level connect
    timeout, which IS TimeoutError since Python 3.10) well within its own
    budget must yield an error card with the RAISED message -- not the
    fabricated scheduler-imposed "widget timed out after Ns" it never hit."""

    async def fn() -> dict:
        raise TimeoutError("connection to api.example.com timed out")

    out = await run_widget(make_widget(fn, timeout=30))
    assert out["error"] == "connection to api.example.com timed out"
    assert "timed out after 30" not in out["error"]  # never the scheduler's message
    assert "data" not in out


async def test_run_widget_timeout_preserves_identity() -> None:
    """Same identity-survives-error guarantee as a raising widget (above),
    for the timeout path specifically."""

    async def fn() -> dict:
        await asyncio.sleep(5)
        return {}

    out = await run_widget(
        make_widget(fn, id="slowwidget", title="Slow Widget", renderer="markdown", timeout=0.05)
    )
    assert out["id"] == "slowwidget"
    assert out["title"] == "Slow Widget"
    assert out["renderer"] == "markdown"
    assert out["error"] == "widget timed out after 0.05s"


async def test_run_widget_preserves_identity_on_error() -> None:
    """Even though a raising widget degrades to error shape, id/title/renderer
    survive so a caller can still say WHICH widget failed."""

    async def fn() -> dict:
        raise ValueError("nope")

    out = await run_widget(make_widget(fn, id="mywidget", title="My Widget", renderer="markdown"))
    assert out["id"] == "mywidget"
    assert out["title"] == "My Widget"
    assert out["renderer"] == "markdown"
    assert out["error"] == "nope"


# --- action resolution -------------------------------------------------------


def _tasks_app() -> App:
    app = App("tasks", description="d")

    @app.tool(action="Done")
    async def complete(id: int) -> str:
        """Complete a task."""
        return f"done {id}"

    @app.tool(action="Refresh")
    async def refresh() -> str:
        """Refresh."""
        return "ok"

    @app.tool(confirm=True, action="Delete")
    async def wipe(id: int) -> str:
        """Delete a task."""
        return "gone"

    return app


def _actions_of(app: App) -> dict[str, Tool]:
    return {name: tool for name, tool in app.tools.items() if tool.action is not None}


async def test_card_carries_its_qualified_id() -> None:
    app = _tasks_app()

    @app.widget(title="T", renderer="markdown")
    async def today() -> str:
        return "hi"

    card = await run_widget(app.widgets["today"], _actions_of(app))
    assert card["qualified_id"] == "tasks.today"
    assert card["id"] == "today"


async def test_item_action_resolves_label_confirm_and_coerced_args() -> None:
    app = _tasks_app()

    @app.widget(title="T", renderer="list")
    async def today() -> list[dict[str, object]]:
        # The "4" is deliberate: it proves coercion actually ran rather than
        # the raw payload being copied through.
        return [{"title": "Buy milk", "action": {"tool": "complete", "args": {"id": "4"}}}]

    card = await run_widget(app.widgets["today"], _actions_of(app))
    action = card["data"][0]["action"]
    assert action == {"tool": "complete", "args": {"id": 4}, "label": "Done", "confirm": False}


async def test_item_label_overrides_the_tools_label() -> None:
    app = _tasks_app()

    @app.widget(title="T", renderer="list")
    async def today() -> list[dict[str, object]]:
        return [
            {
                "title": "Buy milk",
                "action": {"tool": "complete", "args": {"id": 4}, "label": "Undo"},
            }
        ]

    card = await run_widget(app.widgets["today"], _actions_of(app))
    assert card["data"][0]["action"]["label"] == "Undo"


async def test_item_action_inherits_the_tools_confirm_flag() -> None:
    app = _tasks_app()

    @app.widget(title="T", renderer="list")
    async def today() -> list[dict[str, object]]:
        return [{"title": "Buy milk", "action": {"tool": "wipe", "args": {"id": 4}}}]

    card = await run_widget(app.widgets["today"], _actions_of(app))
    assert card["data"][0]["action"]["confirm"] is True


async def test_item_action_naming_an_unknown_tool_degrades_to_an_error_card() -> None:
    app = _tasks_app()

    @app.widget(title="T", renderer="list")
    async def today() -> list[dict[str, object]]:
        return [{"title": "Buy milk", "action": {"tool": "nope", "args": {}}}]

    card = await run_widget(app.widgets["today"], _actions_of(app))
    assert "error" in card
    assert "nope" in card["error"]
    assert card["qualified_id"] == "tasks.today"


async def test_item_action_naming_another_apps_tool_degrades_to_an_error_card() -> None:
    """The mapping handed to run_widget holds only this app's labelled tools,
    so a foreign tool is absent by construction rather than by a check."""
    _tasks_app()
    notes = App("notes", description="d")

    @notes.widget(title="T", renderer="list")
    async def recent() -> list[dict[str, object]]:
        return [{"title": "n", "action": {"tool": "complete", "args": {"id": 1}}}]

    card = await run_widget(notes.widgets["recent"], _actions_of(notes))
    assert "error" in card


async def test_item_action_with_uncoercible_args_degrades_to_an_error_card() -> None:
    app = _tasks_app()

    @app.widget(title="T", renderer="list")
    async def today() -> list[dict[str, object]]:
        return [{"title": "Buy milk", "action": {"tool": "complete", "args": {"id": "abc"}}}]

    card = await run_widget(app.widgets["today"], _actions_of(app))
    assert "error" in card


async def test_card_level_actions_resolve() -> None:
    app = _tasks_app()

    @app.widget(title="T", renderer="markdown", actions=["refresh"])
    async def today() -> str:
        return "hi"

    card = await run_widget(app.widgets["today"], _actions_of(app))
    assert card["actions"] == [
        {"tool": "refresh", "args": {}, "label": "Refresh", "confirm": False}
    ]


async def test_a_widget_with_no_actions_has_an_empty_action_list() -> None:
    app = _tasks_app()

    @app.widget(title="T", renderer="markdown")
    async def today() -> str:
        return "hi"

    card = await run_widget(app.widgets["today"], _actions_of(app))
    assert card["actions"] == []
