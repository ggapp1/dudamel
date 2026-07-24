from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import MetaData

from dudamel.app import App
from dudamel.contract.types import Job, Tool, Widget
from dudamel.exceptions import RegistryError
from dudamel.models_core import CoreBase

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
