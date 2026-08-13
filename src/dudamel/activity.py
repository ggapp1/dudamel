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
    source: str | None = None,
) -> None:
    """Record one tool execution.

    Both `actor` and `source` default to None, which reads as unattributed.
    Every caller names its own surface -- the router included. Defaulting
    `source` to any one surface would make an omission indistinguishable
    from a genuine claim, and a wrong attribution in an audit log is worse
    than a missing one because nothing later can tell the two apart.
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
