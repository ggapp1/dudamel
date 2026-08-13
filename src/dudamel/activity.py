from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from dudamel.db import Database
from dudamel.models_core import Activity

_PREVIEW_CAP = 500


def json_safe(obj: Any) -> Any:
    """Coerced tool args contain Enum members and dates (by contract) —
    this is the one serialization boundary before any JSON column."""
    if isinstance(obj, Enum):
        return json_safe(obj.value)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


async def log_activity(
    db: Database,
    *,
    tool: str,
    args: dict[str, Any],
    status: str,
    result_preview: str | None = None,
    conversation_id: int | None = None,
    actor: str | None = None,
    source: str = "router",
) -> None:
    """Record one tool execution.

    `source` defaults to "router" because that is the only caller that
    predates the deterministic plane and it is, definitionally, the router.
    Surfaces that invoke a tool directly pass their own value.
    """
    async with db.session() as s:
        s.add(
            Activity(
                conversation_id=conversation_id,
                tool=tool,
                args=json_safe(args),
                status=status,
                result_preview=(result_preview or "")[:_PREVIEW_CAP] or None,
                actor=actor,
                source=source,
            )
        )
