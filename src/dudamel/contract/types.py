from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from dudamel.contract.schema import ToolSchema

TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@dataclass
class Tool:
    name: str
    app_name: str
    description: str
    fn: Callable[..., Awaitable[Any]]
    schema: ToolSchema
    read_only: bool
    confirm: bool
    timeout: float
    origin: str = "native"  # "native" | "mcp" (Plan 4)


@dataclass
class Widget:
    id: str
    app_name: str
    title: str
    renderer: str
    fn: Callable[[], Awaitable[Any]]


@dataclass
class Job:
    id: str
    app_name: str
    fn: Callable[[], Awaitable[None]]
    cron: str | None = None
    interval_seconds: int | None = None
    timeout: float = 300.0
    extra: dict = field(default_factory=dict)
