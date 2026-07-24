from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from dudamel.contract.schema import ToolSchema
from dudamel.contract.types import TOOL_NAME_RE, Job, Tool, Widget
from dudamel.exceptions import RegistryError, RuntimeNotBoundError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from dudamel.db import Database

APP_NAME_RE = re.compile(r"^[a-z][a-z0-9]{0,31}$")


class App:
    def __init__(self, name: str, *, description: str) -> None:
        if not APP_NAME_RE.match(name):
            raise RegistryError(
                f"app name {name!r} must start with [a-z] and contain only [a-z0-9];"
                " it prefixes table names"
            )
        self.name = name
        self.description = description
        self.tools: dict[str, Tool] = {}
        self.widgets: dict[str, Widget] = {}
        self.jobs: dict[str, Job] = {}
        self._llm: Callable[..., Awaitable[Any]] | None = None  # bound in Plan 2
        self._notify: Callable[..., Awaitable[None]] | None = None  # bound in Plan 3
        self._database: Any | None = None  # bound in Task 9 tests / run phase

    # --- tools -------------------------------------------------------------
    def tool(
        self,
        fn: Callable | None = None,
        *,
        read_only: bool = False,
        confirm: bool = False,
        timeout: float = 30.0,
    ) -> Callable:
        if fn is not None:  # bare @app.tool
            return self._register_tool(fn, read_only=read_only, confirm=confirm, timeout=timeout)

        def wrap(f: Callable) -> Callable:
            return self._register_tool(f, read_only=read_only, confirm=confirm, timeout=timeout)

        return wrap

    def _register_tool(
        self, fn: Callable, *, read_only: bool, confirm: bool, timeout: float
    ) -> Callable:
        name = fn.__name__
        if not TOOL_NAME_RE.match(name):
            raise RegistryError(f"tool name {name!r} must match {TOOL_NAME_RE.pattern}")
        doc = (inspect.getdoc(fn) or "").strip()
        if not doc:
            raise RegistryError(f"tool {name!r} needs a docstring — it is the LLM's description")
        if name in self.tools:
            raise RegistryError(f"tool {name!r} already registered on app {self.name!r}")
        self.tools[name] = Tool(
            name=name,
            app_name=self.name,
            description=doc,
            fn=fn,
            schema=ToolSchema(fn),
            read_only=read_only,
            confirm=confirm,
            timeout=timeout,
        )
        return fn

    # --- widgets -----------------------------------------------------------
    def widget(self, *, title: str, renderer: str) -> Callable:
        from dudamel.contract.renderers import RENDERERS

        if renderer not in RENDERERS:
            raise RegistryError(f"unknown renderer {renderer!r}; choose one of {sorted(RENDERERS)}")

        def wrap(fn: Callable[[], Awaitable[Any]]) -> Callable:
            wid = fn.__name__
            if wid in self.widgets:
                raise RegistryError(f"widget {wid!r} already registered on app {self.name!r}")
            self.widgets[wid] = Widget(
                id=wid, app_name=self.name, title=title, renderer=renderer, fn=fn
            )
            return fn

        return wrap

    # --- jobs --------------------------------------------------------------
    def job(
        self,
        *,
        cron: str | None = None,
        interval_seconds: int | None = None,
        timeout: float = 300.0,
    ) -> Callable:
        if (cron is None) == (interval_seconds is None):
            raise RegistryError("job needs exactly one of cron= or interval_seconds=")
        if cron is not None:
            from apscheduler.triggers.cron import CronTrigger

            try:
                CronTrigger.from_crontab(cron)
            except ValueError as e:
                raise RegistryError(f"invalid cron expression {cron!r}: {e}") from e

        def wrap(fn: Callable[[], Awaitable[None]]) -> Callable:
            jid = f"{self.name}.{fn.__name__}"
            if jid in self.jobs:
                raise RegistryError(f"job {jid!r} already registered")
            self.jobs[jid] = Job(
                id=jid,
                app_name=self.name,
                fn=fn,
                cron=cron,
                interval_seconds=interval_seconds,
                timeout=timeout,
            )
            return fn

        return wrap

    # --- runtime capabilities ------------------------------------------------
    async def llm(self, *args: Any, **kwargs: Any) -> Any:
        if self._llm is None:
            raise RuntimeNotBoundError("app.llm is bound at run time (Plan 2); not in tests/import")
        return await self._llm(*args, **kwargs)

    async def notify(self, *args: Any, **kwargs: Any) -> None:
        if self._notify is None:
            raise RuntimeNotBoundError("app.notify is bound at run time (Plan 3)")
        await self._notify(*args, **kwargs)

    async def to_thread(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    # --- database ------------------------------------------------------------
    def bind_database(self, database: Database) -> None:
        self._database = database

    def db(self) -> AbstractAsyncContextManager[AsyncSession]:
        if self._database is None:
            raise RuntimeNotBoundError(
                f"app {self.name!r} has no database bound; Orchestrator binds it at run time"
            )
        return self._database.session()
