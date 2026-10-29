"""Widget execution: turns a registered Widget into dashboard-ready JSON.

Thin runner — zero business logic, zero LLM calls, and no way for a widget to
make one: `App.llm()` refuses while `IN_WIDGET` is set. A widget that raises,
times out its own fn(), or returns a payload its renderer rejects must never
take down the whole dashboard: it degrades to an error entry instead of
propagating.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, ValidationError

from dudamel.activity import json_safe
from dudamel.contract.renderers import (
    ItemAction,
    ResolvedAction,
    clean_display_text,
    validate_widget_payload,
)
from dudamel.contract.types import Tool, Widget

logger = logging.getLogger("dudamel.widgets")

# True for exactly the duration of a widget's fn(). `App.llm()` reads it and
# refuses: the dashboard must render with every model tier dead, and "widgets
# don't call the model" is only a guarantee if the framework enforces it. Lives
# here (not in app.py) because widgets.py is the one place that knows when a
# widget is running; app.py imports it, never the reverse.
#
# A ContextVar rather than a plain flag so concurrent renders (the dashboard
# runs widgets in parallel) cannot see each other's state. Because a context is
# copied into any task created from it, a task the widget spawns INHERITS the
# flag -- intended: spawning a task must not be a way around the rule. A thread
# started via `app.to_thread` also copies the context (asyncio.to_thread does),
# but a thread the widget starts by other means does not; that is outside what
# a runtime guard can see.
IN_WIDGET: ContextVar[bool] = ContextVar("in_widget", default=False)


def _to_jsonable(value: Any) -> Any:
    """validate_widget_payload returns pydantic models (stat/table), a list of
    them (list), or a bare str (markdown). Callers of run_widget (API
    responses, Jinja templates) must never see a pydantic instance."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _resolve(spec: ItemAction, app_actions: Mapping[str, Tool]) -> dict[str, Any]:
    """Turn an app-written `ItemAction` into the descriptor a surface renders.

    `app_actions` holds only the action-labelled tools of this widget's own
    app, so an unknown name, an unlabelled tool and another app's tool are one
    indistinguishable failure -- which is the point: same-app ownership is
    enforced by what is in the mapping, not by a rule that could be forgotten.

    Arguments are coerced here so a malformed action becomes a visible error
    card instead of a button that only fails once someone clicks it. This does
    NOT make execution-time validation redundant: arguments arrive back over
    the wire from a browser, and a value is never trusted merely because a
    rendered card once contained it.
    """
    tool = app_actions.get(spec.tool)
    if tool is None:
        raise ValueError(
            f"action {spec.tool!r} is not an action-labelled tool of this widget's app"
        )
    try:
        coerced = tool.schema.arg_model.model_validate(spec.args)
    except ValidationError as e:
        raise ValueError(f"action {spec.tool!r} has invalid arguments: {e}") from e
    return ResolvedAction(
        tool=spec.tool,
        # Coercion yields Enum members and dates by contract; both surfaces
        # serialize this, so it crosses that boundary exactly once, here.
        args=json_safe(coerced.model_dump()),
        label=spec.label or tool.action or spec.tool,
        confirm=tool.confirm,
    ).model_dump()


def _card_title(widget: Widget) -> str:
    """The title both surfaces show, cleaned once here. Telegram strips it on
    its own, but the web renders the card as served, and HTML escaping does
    not remove a bidi override. Falls back to the id if nothing printable is
    left, so a card never has a blank header."""
    return clean_display_text(widget.title) or widget.id


def _error_card(widget: Widget, message: str) -> dict[str, Any]:
    return {
        "id": widget.id,
        "qualified_id": widget.qualified_id,
        "title": _card_title(widget),
        "renderer": widget.renderer,
        # An exception's text is app- or library-authored and can echo
        # synced-in data; cleaned here for the same reason as the title.
        "error": clean_display_text(message),
        "actions": [],
    }


async def run_widget(
    widget: Widget, app_actions: Mapping[str, Tool] | None = None
) -> dict[str, Any]:
    """Run one widget's fn(), validate its payload against its own renderer,
    resolve any actions it declares, and shape the result for rendering.

    `app_actions` maps tool name -> Tool for the action-labelled tools of this
    widget's own app. Passing plain data rather than a registry keeps this
    module free of business logic.

    Success: {"id", "qualified_id", "title", "renderer", "data", "actions"}.
    Failure (fn() raised, timed out, its payload doesn't match the renderer,
    or an action doesn't resolve): the same keys with "error" replacing
    "data" -- identity is kept so a caller can still tell WHICH widget failed
    and still place it in its section.
    """
    actions = app_actions or {}
    try:
        # `asyncio.timeout` lets us tell OUR deadline apart from a TimeoutError
        # the widget raised itself (an OS connect timeout IS TimeoutError since
        # Python 3.10). `cm.expired()` is true only when the context manager's
        # own deadline fired -- a widget-raised TimeoutError falls through to
        # the generic handler below and reports its real message, not a
        # fabricated "widget timed out after Ns".
        async with asyncio.timeout(widget.timeout) as cm:
            # Reset through the token in `finally` so the flag cannot outlive
            # the widget, on success, raise or timeout alike.
            token = IN_WIDGET.set(True)
            try:
                raw = await widget.fn()
            finally:
                IN_WIDGET.reset(token)
    except TimeoutError as e:
        if not cm.expired():
            logger.warning("widget %s (app %s) failed: %s", widget.id, widget.app_name, e)
            return _error_card(widget, str(e))
        message = f"widget timed out after {widget.timeout}s"
        logger.warning("widget %s (app %s) %s", widget.id, widget.app_name, message)
        return _error_card(widget, message)
    except Exception as e:  # widget bugs must not crash render_widgets()
        logger.warning("widget %s (app %s) failed: %s", widget.id, widget.app_name, e)
        return _error_card(widget, str(e))
    try:
        validated = validate_widget_payload(widget.renderer, raw)
        data = _to_jsonable(validated)
        if widget.renderer == "list":
            for item, rendered in zip(validated, data, strict=True):
                if item.action is not None:
                    rendered["action"] = _resolve(item.action, actions)
        card_actions = [_resolve(ItemAction(tool=name), actions) for name in widget.actions]
    except Exception as e:  # payload, or an action, doesn't resolve
        logger.warning("widget %s (app %s) failed: %s", widget.id, widget.app_name, e)
        return _error_card(widget, str(e))
    return {
        "id": widget.id,
        "qualified_id": widget.qualified_id,
        "title": _card_title(widget),
        "renderer": widget.renderer,
        "data": data,
        "actions": card_actions,
    }
