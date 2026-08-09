from __future__ import annotations

from collections.abc import Sequence

from dudamel.app import App
from dudamel.mcp_mount import MCPServerConfig
from dudamel.registry import Registry


class Orchestrator:
    """Pure registration. All side effects (db, scheduler, bot, MCP) happen in
    the run phase (`dudamel run`) — never at construction, so that
    `dudamel db migrate` and tests can import a project safely."""

    def __init__(self, apps: Sequence[App] = (), mcp: Sequence[str | MCPServerConfig] = ()) -> None:
        self.mcp: list[str | MCPServerConfig] = list(mcp)
        self.registry = Registry(apps)
