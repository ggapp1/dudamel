from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import MetaData

from dudamel.app import App
from dudamel.contract.types import Job, Tool, Widget
from dudamel.exceptions import RegistryError

RESERVED_APP_NAMES = {"dudamel", "core"}


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
