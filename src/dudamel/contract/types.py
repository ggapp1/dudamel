from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

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
    origin: Literal["native", "mcp"] = "native"  # "mcp" = mounted via mcp_mount.py
    # Whether this tool's return value may contain text an attacker controls.
    # Orthogonal to `read_only`: a read-only fetch is never gated itself --
    # search and fetch have to stay frictionless -- but it taints the turn, so
    # a later mutation stops and asks.
    external: bool = False

    @property
    def untrusted(self) -> bool:
        """Whether this tool's output must be treated as attacker-influenceable.

        The single predicate the taint system reads. A property rather than a
        stored flag because `origin` is assigned after construction in real
        code paths (a reconnect force-gating a tool in place), which any
        value computed once at construction would miss.
        """
        return self.external or self.origin == "mcp"


@dataclass
class Widget:
    id: str
    app_name: str
    title: str
    renderer: str
    fn: Callable[[], Awaitable[Any]]
    timeout: float = 15.0


@dataclass
class Job:
    id: str
    app_name: str
    fn: Callable[[], Awaitable[None]]
    cron: str | None = None
    interval_seconds: int | None = None
    timeout: float = 300.0
