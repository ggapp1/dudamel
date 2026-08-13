from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from pydantic import AliasChoices, AliasPath, BaseModel, ValidationError

from dudamel.contract.renderers import RENDERERS
from dudamel.contract.schema import ToolSchema
from dudamel.contract.types import TOOL_NAME_RE, Job, Tool, Widget
from dudamel.exceptions import AppSettingsError, RegistryError, RuntimeNotBoundError

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncSession

    from dudamel.db import Database

APP_NAME_RE = re.compile(r"^[a-z][a-z0-9]{0,31}$")


class _EmptySettings(BaseModel):
    """Bound marker for an app that declares no settings model."""


_EMPTY_SETTINGS = _EmptySettings()


def _accepted_keys(model: type[BaseModel]) -> set[str]:
    """The key spellings `model.model_validate` will actually accept.

    An alias replaces the field name rather than adding to it: with a plain
    `Field(alias="key")` pydantic accepts `key` and rejects `api_key`, unless
    the model opts into population by name. Listing both would advertise a
    spelling that validation then rejects, so the two sets are kept in step.
    """
    config = model.model_config
    by_name = bool(config.get("validate_by_name") or config.get("populate_by_name"))
    keys: set[str] = set()
    for field_name, field in model.model_fields.items():
        # pydantic mirrors a plain `alias=` into `validation_alias`, so the
        # first branch covers both spellings; `alias` remains the fallback for
        # a field carrying only the serialization-side name.
        if field.validation_alias is not None:
            aliased = True
            sources: list[Any] = (
                list(field.validation_alias.choices)
                if isinstance(field.validation_alias, AliasChoices)
                else [field.validation_alias]
            )
        elif field.alias is not None:
            aliased = True
            sources = [field.alias]
        else:
            aliased = False
            sources = []
        for source in sources:
            if isinstance(source, str):
                keys.add(source)
            elif isinstance(source, AliasPath) and source.path and isinstance(source.path[0], str):
                # a path alias is looked up under its first path element
                keys.add(source.path[0])
        # An alias whose top-level key we cannot name still means the field name
        # is not accepted. Falling back to it would advertise a spelling that
        # validation rejects, which is the failure this function exists to
        # prevent; refusing a key we cannot name fails closed instead.
        if by_name or not aliased:
            keys.add(field_name)
    return keys


class App:
    def __init__(
        self,
        name: str,
        *,
        description: str,
        settings: type[BaseModel] | None = None,
    ) -> None:
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
        self._llm: Callable[..., Awaitable[Any]] | None = None  # bound by Runtime at run time
        self._notify: Callable[..., Awaitable[None]] | None = None  # bound by Runtime at run time
        self._database: Any | None = None  # bound by tests, or by Runtime at run time
        self._model_base: type | None = None
        self.settings_model = settings
        self._settings: BaseModel | None = None  # bound by config resolution, not at import

    # --- tools -------------------------------------------------------------
    def tool(
        self,
        fn: Callable | None = None,
        *,
        read_only: bool = False,
        confirm: bool = False,
        external: bool = False,
        timeout: float = 30.0,
    ) -> Callable:
        if fn is not None:  # bare @app.tool
            return self._register_tool(
                fn, read_only=read_only, confirm=confirm, external=external, timeout=timeout
            )

        def wrap(f: Callable) -> Callable:
            return self._register_tool(
                f, read_only=read_only, confirm=confirm, external=external, timeout=timeout
            )

        return wrap

    def _register_tool(
        self,
        fn: Callable,
        *,
        read_only: bool,
        confirm: bool,
        timeout: float,
        external: bool = False,
    ) -> Callable:
        name = fn.__name__
        if not inspect.iscoroutinefunction(fn):
            raise RegistryError(f"tool {name!r} must be async (define it with `async def`)")
        if not TOOL_NAME_RE.match(name):
            raise RegistryError(f"tool name {name!r} must match {TOOL_NAME_RE.pattern}")
        doc = (inspect.getdoc(fn) or "").strip()
        if not doc:
            raise RegistryError(f"tool {name!r} needs a docstring — it is the LLM's description")
        if name in self.tools:
            raise RegistryError(f"tool {name!r} already registered on app {self.name!r}")
        try:
            schema = ToolSchema(fn)
        except TypeError as e:
            # ToolSchema signature problems (missing type hint, *args/**kwargs,
            # unsupported parameter type, ...) are registration failures like
            # any other here — fold them into the same RegistryError taxonomy
            # instead of leaking a bare TypeError through the decorator.
            raise RegistryError(str(e)) from e
        self.tools[name] = Tool(
            name=name,
            app_name=self.name,
            description=doc,
            fn=fn,
            schema=schema,
            read_only=read_only,
            confirm=confirm,
            external=external,
            timeout=timeout,
        )
        return fn

    # --- widgets -----------------------------------------------------------
    def widget(self, *, title: str, renderer: str, timeout: float = 15.0) -> Callable:
        if renderer not in RENDERERS:
            raise RegistryError(f"unknown renderer {renderer!r}; choose one of {sorted(RENDERERS)}")

        def wrap(fn: Callable[[], Awaitable[Any]]) -> Callable:
            wid = fn.__name__
            if not inspect.iscoroutinefunction(fn):
                raise RegistryError(f"widget {wid!r} must be async (define it with `async def`)")
            required = [
                p.name
                for p in inspect.signature(fn).parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]
            if required:
                raise RegistryError(
                    f"widget {wid!r} must take no arguments; found required parameter(s) {required}"
                )
            if wid in self.widgets:
                raise RegistryError(f"widget {wid!r} already registered on app {self.name!r}")
            self.widgets[wid] = Widget(
                id=wid, app_name=self.name, title=title, renderer=renderer, fn=fn, timeout=timeout
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
            if not inspect.iscoroutinefunction(fn):
                raise RegistryError(f"job {jid!r} must be async (define it with `async def`)")
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
            raise RuntimeNotBoundError(
                "app.llm is bound at run time by Runtime; not in tests/import"
            )
        return await self._llm(*args, **kwargs)

    async def notify(self, *args: Any, **kwargs: Any) -> None:
        if self._notify is None:
            raise RuntimeNotBoundError("app.notify is bound at run time by Runtime")
        await self._notify(*args, **kwargs)

    async def to_thread(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    # --- settings ------------------------------------------------------------
    @property
    def settings(self) -> Any:
        """This app's validated settings. Bound during config resolution, not at
        import time -- reading it before then is a wiring bug, so it fails
        loudly instead of handing back a None to trip over later."""
        if self._settings is None:
            raise RuntimeNotBoundError(f"app {self.name!r}: settings accessed before load")
        return self._settings

    def bind_settings(self, values: Mapping[str, Any]) -> None:
        """Validate a raw config block for this app and bind the result.

        Unknown keys are rejected here rather than by the model's own
        `extra="forbid"`: pydantic ignores extras by default, so leaving this to
        app authors would make a typo'd setting silently do nothing.
        """
        if self.settings_model is None:
            if values:
                raise AppSettingsError(
                    f"app {self.name!r} takes no settings, got: " + ", ".join(sorted(values))
                )
            self._settings = _EMPTY_SETTINGS
            return
        known = _accepted_keys(self.settings_model)
        unknown = sorted(set(values) - known)
        if unknown:
            raise AppSettingsError(
                f"app {self.name!r}: unknown setting(s) {', '.join(unknown)}; "
                f"known: {', '.join(sorted(known))}"
            )
        try:
            self._settings = self.settings_model.model_validate(values)
        except ValidationError as e:
            raise AppSettingsError(f"app {self.name!r}: {e}") from e

    # --- database ------------------------------------------------------------
    def bind_database(self, database: Database) -> None:
        self._database = database

    def db(self) -> AbstractAsyncContextManager[AsyncSession]:
        if self._database is None:
            raise RuntimeNotBoundError(
                f"app {self.name!r} has no database bound; Orchestrator binds it at run time"
            )
        return self._database.session()

    # --- model sugar ---------------------------------------------------------
    @property
    def Model(self) -> type:  # noqa: N802 — class-like property, PEP8 exception intended
        if self._model_base is None:
            from dudamel.modelsugar import make_model_base

            self._model_base = make_model_base(self.name)
        return self._model_base

    @property
    def metadata(self) -> MetaData:
        return self.Model.metadata

    @staticmethod
    def now() -> object:
        from dudamel.modelsugar import NOW

        return NOW
