from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy import MetaData

from dudamel.app import App
from dudamel.contract.types import TOOL_NAME_RE, Job, Tool, Widget
from dudamel.exceptions import RegistryError
from dudamel.models_core import CoreBase

logger = logging.getLogger("dudamel.registry")

RESERVED_APP_NAMES = {"dudamel", "core"}

# Every string an app's table prefix must never be a prefix of: the framework's
# own core tables (conversations, messages, ...) plus the alembic version-table
# namespace. Derived from CoreBase.metadata rather than hardcoded so a future
# core table automatically extends the forbidden-prefix set.
_CORE_NAMESPACE_CANDIDATES = frozenset(CoreBase.metadata.tables) | {"alembic_"}


def _core_namespace_collision(app_name: str) -> str | None:
    """Return the core identifier an app's table prefix would shadow, if any.

    An app named "job" produces prefix "job_", which is a prefix of the core
    table "job_runs" — tables reflected under that name would silently be
    treated as belonging to the "job" app by migrate.py's prefix allowlist.
    Checking prefix-of (not equality) also catches "alembic" against the
    "alembic_version_core"/"alembic_version_apps" version tables via the
    "alembic_" sentinel.
    """
    prefix = f"{app_name}_"
    for candidate in _CORE_NAMESPACE_CANDIDATES:
        if candidate.startswith(prefix):
            return candidate
    return None


class Registry:
    def __init__(self, apps: Sequence[App]) -> None:
        self.apps: dict[str, App] = {}
        self.tools: dict[str, Tool] = {}
        self.widgets: list[Widget] = []
        self.jobs: list[Job] = []
        self.metadatas: dict[str, MetaData] = {}

        for app in apps:
            if app.name in RESERVED_APP_NAMES:
                raise RegistryError(f"app name {app.name!r} is reserved")
            collision = _core_namespace_collision(app.name)
            if collision is not None:
                raise RegistryError(
                    f"app name {app.name!r} produces table prefix {app.name}_ "
                    f"which collides with the core namespace (shadows {collision!r}); "
                    "choose a different app name"
                )
            if app.name in self.apps:
                raise RegistryError(f"duplicate app name {app.name!r}")
            self.apps[app.name] = app
            for name, tool in app.tools.items():
                if name in self.tools:
                    raise RegistryError(
                        f"tool name {name!r} registered by both "
                        f"{self.tools[name].app_name!r} and {app.name!r}; rename one"
                    )
                self.tools[name] = tool
            self.widgets.extend(app.widgets.values())
            self.jobs.extend(app.jobs.values())
            self.metadatas[app.name] = app.metadata

    def add_mcp_tools(self, tools: Sequence[Tool]) -> None:
        """Sanctioned entry point for grafting MCP-discovered tools in after
        construction -- `Runtime.start()` is the only caller.

        Unlike `MCPMount.mount()`'s per-server "warn and skip" degradation for
        environmental failures (server unreachable, handshake failure), a
        malformed name or a collision with a NATIVE tool here is a
        configuration bug the operator must fix, so it raises `RegistryError`
        immediately and mounts nothing from this batch -- both are checked
        before anything is added, so a batch either fully succeeds or fully
        fails.

        A collision with another MCP-origin tool (already registered, or
        earlier in this same batch -- `MCPMount.mount()` already dedupes its
        own output this way, but this is the sanctioned entry point and
        defends independently) is NOT a configuration bug an operator can
        fix: it can come from an untrusted MCP server spoofing another's
        identity, or two long tool names truncating to the same 64-char
        name. Raising there would let one hostile/accidental MCP server
        take the whole batch (or, via `Runtime.start()`, the process) down.
        So that case is a WARN + drop (first mount wins) instead.
        """
        seen: dict[str, Tool] = {}
        for tool in tools:
            if tool.origin != "mcp":
                raise RegistryError(
                    f"add_mcp_tools received a non-mcp tool {tool.name!r} (origin={tool.origin!r})"
                )
            if not TOOL_NAME_RE.match(tool.name):
                raise RegistryError(
                    f"mcp tool name {tool.name!r} must match {TOOL_NAME_RE.pattern}"
                )
            existing = self.tools.get(tool.name) or seen.get(tool.name)
            if existing is not None:
                if existing.origin != "mcp":
                    raise RegistryError(
                        f"mcp tool name {tool.name!r} (from {tool.app_name!r}) collides with an "
                        f"existing {existing.origin} tool registered by {existing.app_name!r}; "
                        "rename the MCP tool or the native one"
                    )
                logger.warning(
                    "mcp: tool %r from %r collides with an existing mcp tool from %r -- "
                    "dropping this one (first mount wins)",
                    tool.name,
                    tool.app_name,
                    existing.app_name,
                )
                continue
            seen[tool.name] = tool
        self.tools.update(seen)
