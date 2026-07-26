from __future__ import annotations

from collections.abc import Sequence

from dudamel.app import App
from dudamel.registry import Registry


class Orchestrator:
    """Pure registration. All side effects (db, scheduler, bot, MCP) happen in
    the run phase (`dudamel run`) — never at construction, so that
    `dudamel db migrate` and tests can import a project safely."""

    def __init__(self, apps: Sequence[App] = (), mcp: Sequence[str] = ()) -> None:
        self.mcp: list[str] = list(mcp)
        self.registry = Registry(apps)
