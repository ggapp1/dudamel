"""Widget execution: turns a registered Widget into dashboard-ready JSON.

Thin runner — zero business logic, zero LLM calls. A widget that raises,
times out its own fn(), or returns a payload its renderer rejects must never
take down the whole dashboard: it degrades to an error entry instead of
propagating.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel

from dudamel.contract.renderers import validate_widget_payload
from dudamel.contract.types import Widget

logger = logging.getLogger("dudamel.widgets")


def _to_jsonable(value: Any) -> Any:
    """validate_widget_payload returns pydantic models (stat/table), a list of
    them (list), or a bare str (markdown). Callers of run_widget (API
    responses, Jinja templates) must never see a pydantic instance."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _error_card(widget: Widget, message: str) -> dict[str, Any]:
    return {
        "id": widget.id,
        "title": widget.title,
        "renderer": widget.renderer,
        "error": message,
    }


async def run_widget(widget: Widget) -> dict[str, Any]:
    """Run one widget's fn(), validate its payload against its own renderer,
    and shape the result for rendering.

    Success: {"id", "title", "renderer", "data"}.
    Failure (fn() raised, timed out, or its payload doesn't match the
    renderer): {"id", "title", "renderer", "error"} — identity is kept so a
    caller can still tell WHICH widget failed.
    """
    try:
        raw = await asyncio.wait_for(widget.fn(), timeout=widget.timeout)
    except TimeoutError:
        message = f"widget timed out after {widget.timeout}s"
        logger.warning("widget %s (app %s) %s", widget.id, widget.app_name, message)
        return _error_card(widget, message)
    except Exception as e:  # widget bugs must not crash render_widgets()
        logger.warning("widget %s (app %s) failed: %s", widget.id, widget.app_name, e)
        return _error_card(widget, str(e))
    try:
        data = _to_jsonable(validate_widget_payload(widget.renderer, raw))
    except Exception as e:  # payload doesn't match its own renderer's schema
        logger.warning("widget %s (app %s) failed: %s", widget.id, widget.app_name, e)
        return _error_card(widget, str(e))
    return {
        "id": widget.id,
        "title": widget.title,
        "renderer": widget.renderer,
        "data": data,
    }
