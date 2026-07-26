"""Acceptance tests for dudamel/widgets.py::run_widget, including its
timeout enforcement."""

import asyncio

from dudamel.contract.types import Widget
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
        "title": "W",
        "renderer": "stat",
        "data": {"label": "Vol", "value": 12, "unit": "kg", "delta": None},
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
    assert out["data"] == [{"title": "t", "subtitle": None, "url": "http://x"}]


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
