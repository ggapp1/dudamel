"""Composition root. Interfaces call chat()/resolve_confirmation() and
nothing else. Construction is in-memory only; start() touches the world."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from dudamel.activity import json_safe, log_activity
from dudamel.compaction import Compactor
from dudamel.config import Settings, TierConfig
from dudamel.contract.types import Tool
from dudamel.convo import ConversationStore
from dudamel.db import Database
from dudamel.exceptions import (
    ActionArgumentError,
    DudamelError,
    LLMError,
    RegistryError,
    ToolValidationError,
    UnknownActionError,
)
from dudamel.llm.anthropic import AnthropicProvider
from dudamel.llm.client import LLMClient, Tier
from dudamel.llm.openai_compat import OpenAICompatProvider
from dudamel.llm.prompted_tools import PromptedToolsProvider
from dudamel.llm.provider import Provider
from dudamel.llm.types import Message
from dudamel.mcp_mount import MCPMount
from dudamel.migrate import pending_migrations, upgrade_all, upgrade_core
from dudamel.models_core import Activity, Conversation, JobRun, PendingConfirmation
from dudamel.orchestrator import Orchestrator
from dudamel.router import ChatReply, Router
from dudamel.scheduler import JobScheduler
from dudamel.widgets import run_widget

logger = logging.getLogger("dudamel.runtime")


def _build_raw_provider(name: str, cfg: TierConfig) -> Provider:
    if cfg.provider == "openai-compatible":
        if not cfg.base_url:
            raise RegistryError(f"tier {name!r}: openai-compatible provider needs base_url")
        key = os.environ.get(cfg.api_key_env, "unused") if cfg.api_key_env else "unused"
        return OpenAICompatProvider(base_url=cfg.base_url, api_key=key)
    if cfg.provider == "anthropic":
        env = cfg.api_key_env or "ANTHROPIC_API_KEY"
        key = os.environ.get(env)
        if not key:
            raise RegistryError(f"tier {name!r}: environment variable {env} is not set")
        return AnthropicProvider(api_key=key)
    # "fake"
    raise RegistryError(
        f"tier {name!r}: provider 'fake' requires a providers= override "
        "(dudamel.llm.testing.FakeProvider)"
    )


def build_provider(name: str, cfg: TierConfig) -> Provider:
    """Construct the Provider a tier's config names -- shared by
    Runtime._build_tiers (real runs) and `dudamel doctor --probe-tools`
    (which needs a live Provider without standing up a whole Runtime).

    Wraps in PromptedToolsProvider when `cfg.tool_calling == "prompted"`, so
    the probe exercises the same object the router will actually call.
    Probing the raw backend of a prompted tier would only ever reconfirm
    that native tool calling is absent -- which the operator already told
    us by setting the config -- and would say nothing about whether the
    prompted fallback itself can round-trip a call for this model.
    """
    provider = _build_raw_provider(name, cfg)
    if cfg.tool_calling == "prompted":
        provider = PromptedToolsProvider(provider)
    return provider


class Runtime:
    def __init__(
        self,
        orchestrator: Orchestrator,
        settings: Settings,
        *,
        providers: dict[str, Provider] | None = None,
        suite_lanes: Sequence[tuple[str, Path]] = (),
    ) -> None:
        self._settings = settings
        # (app name, versions dir) for each enabled suite app. Resolution owns
        # which apps those are; Runtime only needs their lanes, so that start()
        # applies and gates on exactly the migration tiers that will run.
        self._suite_lanes = list(suite_lanes)
        self._db = Database(settings.database_url)
        self._registry = orchestrator.registry
        self._mcp_commands = list(orchestrator.mcp)
        self._mcp_mount: MCPMount | None = None
        tiers = self._build_tiers(providers or {})
        self._llm = LLMClient(tiers=tiers, db=self._db, budget=settings.llm_budget)
        self._convo = ConversationStore(self._db)
        self._compactor = self._build_compactor(settings, tiers)
        self._router = Router(
            llm=self._llm,
            registry=self._registry,
            convo=self._convo,
            db=self._db,
            config=settings.router,
            compactor=self._compactor,
        )
        # Created here so Runtime construction (safe to do in tests / at
        # import-adjacent time) fully wires the process, but NOT started:
        # only the single-process assembly (dudamel.serve.serve) calls
        # scheduler.start(), once everything else (db migrations,
        # interfaces) is up.
        self.scheduler = JobScheduler(self._registry, self._db)
        for app in orchestrator.registry.apps.values():
            app.bind_database(self._db)
            app._llm = self._make_app_llm()
            app._notify = self._fallback_notify

    def _build_tiers(self, overrides: dict[str, Provider]) -> dict[str, Tier]:
        tiers: dict[str, Tier] = {}
        for name, cfg in self._settings.llm_tiers.items():
            if name in overrides:
                provider: Provider = overrides[name]
                # build_provider() already wraps for a "prompted" tier, but
                # an override bypasses build_provider entirely (that's the
                # whole point -- FakeProvider stands in for a real backend
                # in tests), so the wrap has to be applied here too. This is
                # what lets a test drive an end-to-end prompted turn: script
                # a FakeProvider emitting the prompted JSON envelope, wrap
                # it, and assert the router receives real ToolCalls.
                if cfg.tool_calling == "prompted":
                    provider = PromptedToolsProvider(provider)
            else:
                provider = build_provider(name, cfg)
            tiers[name] = Tier(
                name=name, provider=provider, model=cfg.model, max_tokens=cfg.max_tokens
            )
        for name in overrides:
            if name not in tiers:
                # An override for a tier absent from config -- `tiers` already
                # holds every configured tier, so there is no config-backed
                # model to read here; the name exists only as an override.
                tiers[name] = Tier(
                    name=name, provider=overrides[name], model="override", max_tokens=1024
                )
        return tiers

    def _build_compactor(self, settings: Settings, tiers: dict[str, Tier]) -> Compactor | None:
        """`[router] compact_dropped_turns` is opt-in and off by default, so
        this is where its config is validated -- `tiers` (which tier names
        are actually configured) is only known here, at Runtime construction,
        never at Settings/RouterConfig parse time."""
        if not settings.router.compact_dropped_turns:
            return None
        tier_name = settings.router.compaction_tier
        if not tier_name:
            raise RegistryError(
                "[router] compact_dropped_turns is true but compaction_tier is not set"
            )
        if tier_name not in tiers:
            raise RegistryError(
                f"unknown tier {tier_name!r}; configured tiers: {sorted(tiers) or 'none'}"
            )
        return Compactor(llm=self._llm, db=self._db, tier=tier_name)

    def _make_app_llm(self) -> Callable[..., Awaitable[str | dict[str, Any]]]:
        async def app_llm(
            prompt: str | list[Message],
            *,
            tier: str = "standard",
            schema: dict[str, Any] | None = None,
        ) -> str | dict[str, Any]:
            messages = [Message(role="user", text=prompt)] if isinstance(prompt, str) else prompt
            completion = await self._llm.complete(messages, tier=tier, json_schema=schema)
            if schema is None:
                return completion.message.text
            try:
                return json.loads(completion.message.text)
            except json.JSONDecodeError as e:
                raise LLMError(
                    f"model output was not valid JSON for the requested schema: {e}"
                ) from e

        return app_llm

    @staticmethod
    async def _fallback_notify(text: str) -> None:
        logger.warning("notify (no channel configured): %s", text)

    async def start(self) -> None:
        url = self._settings.database_url
        # App migrations live in the PROJECT directory (where `dudamel new`/
        # `dudamel db migrate` create and read them), NOT data_dir. Resolving
        # from project_dir keeps this in lockstep with the CLI so the
        # auto_migrate gate below cannot be hollowed out by a data_dir that
        # differs from the project root.
        project_dir = self._settings.project_dir
        # A programmatic embedder who builds Settings(data_dir=X) directly
        # (not via Settings.load) and drops app migrations under data_dir
        # gets them silently ignored -- resolution is from project_dir, and
        # with auto_migrate=False the gate below sees no pending app
        # migrations and passes. Warn so this isn't a silent no-op.
        if (
            (self._settings.data_dir / "migrations").exists()
            and self._settings.data_dir != project_dir
            and not (project_dir / "migrations").exists()
        ):
            logger.warning(
                "app migrations found under data_dir (%s) but are resolved from "
                "project_dir (%s), which has none -- they will be ignored; move "
                "migrations/ under project_dir or set project_dir to data_dir",
                self._settings.data_dir,
                project_dir,
            )
        if self._settings.auto_migrate:
            await asyncio.to_thread(upgrade_core, url)
            # Not gated on migrations/ alone: a project with no lane of its own
            # can still have enabled suite apps whose lanes must be applied.
            # Skipped entirely when there is nothing to apply: `upgrade_core`
            # above already took a backup, and `upgrade_all` takes a second one
            # of its own, so a project with no lane at all would otherwise pay
            # for a redundant backup on every start.
            if self._suite_lanes or (project_dir / "migrations").exists():
                await asyncio.to_thread(upgrade_all, url, project_dir, self._suite_lanes)
        else:
            pending = await asyncio.to_thread(
                pending_migrations, url, project_dir, self._suite_lanes
            )
            if pending:
                raise DudamelError(
                    "refusing to start: "
                    + "; ".join(pending)
                    + " — auto_migrate is off, so run `dudamel db migrate -m <message>` "
                    "and restart"
                )
        if self._mcp_commands:
            # EXPERIMENTAL: a broken/unreachable MCP server degrades to a
            # warning and zero tools from it (MCPMount.mount()'s job) — it
            # must never make start() fail. A name collision with a native
            # (or another mcp) tool is the one thing that DOES raise here
            # (Registry.add_mcp_tools): that's a configuration bug, not
            # environmental flakiness.
            self._mcp_mount = MCPMount(
                self._mcp_commands,
                env_passthrough=self._settings.mcp.env_passthrough,
                call_timeout=self._settings.mcp.call_timeout,
                mount_timeout=self._settings.mcp.mount_timeout,
                reconnect_attempts=self._settings.mcp.reconnect_attempts,
                reconnect_backoff_seconds=self._settings.mcp.reconnect_backoff_seconds,
                reconnect_cooldown_seconds=self._settings.mcp.reconnect_cooldown_seconds,
            )
            tools = await self._mcp_mount.mount()
            if tools:
                self._registry.add_mcp_tools(tools)
                self._router.refresh_tool_specs()

    async def chat(
        self,
        channel: str,
        text: str,
        *,
        user_id: str,
        client_msg_id: str | None = None,
    ) -> ChatReply:
        return await self._router.handle(
            channel=channel, text=text, user_id=user_id, client_msg_id=client_msg_id
        )

    async def resolve_confirmation(
        self, confirmation_id: str, *, approved: bool, user_id: str
    ) -> ChatReply:
        return await self._router.resolve_confirmation(
            confirmation_id, approved=approved, user_id=user_id
        )

    async def list_pending_confirmations(
        self, channel: str | None = None, *, include_expired: bool = True
    ) -> list[dict[str, Any]]:
        """Pending confirmations, newest last. Confirmations expire lazily --
        the router only flips a past-TTL row to "expired" when the conversation
        is next touched (router.py) -- so a row can sit at status="pending"
        indefinitely after its `expires_at`.

        `include_expired=False` filters those out, for callers that must not
        present a dead confirmation as actionable (the dashboard chat page's
        approve/deny buttons -- clicking an expired one only yields "that
        confirmation expired; nothing was done"). `/api/pending` keeps the
        default True: it lists expired entries deliberately and marks them
        `resolvable=False` for its external cross-channel consumers.
        """
        stmt = (
            select(PendingConfirmation, Conversation.channel)
            .join(Conversation, Conversation.id == PendingConfirmation.conversation_id)
            .where(PendingConfirmation.status == "pending")
        )
        if not include_expired:
            # Boundary matches /api/pending's `resolvable` (expires_at >= now):
            # a row exactly at `now` is still live.
            now = datetime.now(UTC).replace(tzinfo=None)
            stmt = stmt.where(PendingConfirmation.expires_at >= now)
        if channel is not None:
            stmt = stmt.where(Conversation.channel == channel)
        stmt = stmt.order_by(PendingConfirmation.created_at)
        async with self._db.session() as s:
            rows = (await s.execute(stmt)).all()
        return [
            {
                "id": r.id,
                "tool": r.tool,
                "args": r.args,
                "created_at": r.created_at,
                "expires_at": r.expires_at,
                "channel": conv_channel,
                "user_id": r.user_id,
            }
            for r, conv_channel in rows
        ]

    def bind_notify(self, fn: Callable[[str], Awaitable[None]]) -> None:
        """Rebind every app's app.notify() to fn — used by the single-process
        assembly (dudamel.serve.serve) once an interface (e.g. Telegram) is
        up, replacing the WARN-log fallback bound at construction time."""
        for app in self._registry.apps.values():
            app._notify = fn

    async def db_ping(self) -> None:
        """Cheap DB liveness probe for GET /health — touches the database
        without any business logic; raises if the DB is down."""
        async with self._db.session() as s:
            await s.execute(select(1))

    def _app_actions(self, app_name: str) -> dict[str, Tool]:
        """The action-labelled tools of one app, for widget rendering.

        Scoped to a single app deliberately: this is what makes a per-item
        action naming another app's tool impossible rather than merely
        forbidden.
        """
        app = self._registry.apps.get(app_name)
        if app is None:
            return {}
        return {name: tool for name, tool in app.tools.items() if tool.action is not None}

    async def render_widgets(self) -> list[dict[str, Any]]:
        """Run every registered widget concurrently. Data-plane guarantee: no
        model is ever invoked here (widgets.run_widget calls only widget.fn())."""
        return list(
            await asyncio.gather(
                *(
                    run_widget(widget, self._app_actions(widget.app_name))
                    for widget in self._registry.widgets
                )
            )
        )

    async def _log_action_error(
        self, tool_name: str, args: dict[str, Any], detail: str, *, actor: str, source: str
    ) -> None:
        await log_activity(
            self._db,
            tool=tool_name,
            args=args,
            status="error",
            result_preview=detail,
            actor=actor,
            source=source,
        )

    async def run_action(
        self, tool_name: str, args: dict[str, Any], *, actor: str, source: str
    ) -> str:
        """Execute one operator-invoked tool on the deterministic plane.

        Deliberately NOT routed through `Router`: there is no model in this
        path, so there is no window to build, no turn to taint, and no
        confirmation to obtain -- the operator clicking a button IS the human
        decision the confirm machine exists to get.

        Raises UnknownActionError if `tool_name` is not an action-labelled
        tool, ActionArgumentError (a ValueError) if `args` do not coerce, and
        whatever the tool raises otherwise -- except that OUR deadline
        expiring raises a TimeoutError that says so.

        Both of those are typed rather than the bare `KeyError`/`ValueError`
        the two conditions would naturally produce, and for the same reason:
        a tool body raising either one is a tool failure, and a caller that
        could not tell it apart from a lookup or coercion failure would
        misreport a tool that actually ran.

        Every outcome past the name lookup writes exactly one activity row,
        the refused-arguments one included: this is an authenticated,
        state-changing entry point reachable from more than one surface, so
        argument probing against it must not be the one thing the audit log
        cannot see.
        """
        tool = self._registry.tools.get(tool_name)
        if tool is None or tool.action is None:
            raise UnknownActionError(f"no action {tool_name!r}")
        try:
            # The registry's own coercion, not a bare model_validate +
            # model_dump: `validate` returns attribute values, so a nested
            # model parameter reaches the tool as the model instance the tool
            # annotated, where model_dump() would hand it a plain dict. Same
            # coercion the model-facing path uses, so one tool cannot receive
            # two different shapes of argument depending on who invoked it.
            kwargs = tool.schema.validate(args)
        except ToolValidationError as e:
            await log_activity(
                self._db,
                # The raw args, not `kwargs`: coercion is exactly what failed,
                # so what was actually submitted is the only thing to record.
                tool=tool_name,
                args=args,
                status="error",
                result_preview=str(e),
                actor=actor,
                source=source,
            )
            raise ActionArgumentError(str(e)) from e
        try:
            # `asyncio.timeout`, not `wait_for`: `cm.expired()` is true only
            # when OUR deadline fired, so a tool raising TimeoutError itself
            # (an OS connect timeout IS TimeoutError since 3.10) keeps its own
            # message instead of being reported as a fabricated deadline. The
            # same idiom widgets.py and scheduler.py use.
            async with asyncio.timeout(tool.timeout) as cm:
                result = await tool.fn(**kwargs)
        except TimeoutError as e:
            if cm.expired():
                detail = f"action {tool_name} timed out after {tool.timeout}s"
                await self._log_action_error(tool_name, kwargs, detail, actor=actor, source=source)
                raise TimeoutError(detail) from e
            await self._log_action_error(
                tool_name, kwargs, str(e) or type(e).__name__, actor=actor, source=source
            )
            raise
        except Exception as e:
            await self._log_action_error(
                tool_name, kwargs, str(e) or type(e).__name__, actor=actor, source=source
            )
            raise
        # The same normalization the model-facing path uses, so one tool
        # cannot present two different shapes depending on who called it.
        text = result if isinstance(result, str) else json.dumps(json_safe(result))
        cap = self._settings.router.tool_result_cap
        if len(text) > cap:
            # Marker format copied verbatim from window.truncate_tool_result,
            # which can't be reused here: it takes and returns a `Message`.
            # Silent truncation would leave an operator reading output that
            # ends mid-word with no way to tell that from the tool's own
            # output.
            text = text[:cap] + f"…[truncated {len(text) - cap} chars]"
        await log_activity(
            self._db,
            tool=tool_name,
            args=kwargs,
            status="ok",
            result_preview=text,
            actor=actor,
            source=source,
        )
        return text

    async def recent_messages(self, channel: str, limit: int = 200) -> list[dict[str, Any]]:
        """Chat history for `channel` — used by the dashboard's /chat page.
        Creates the conversation if it doesn't exist yet (mirrors chat()'s
        read-or-create); an empty history is a normal result, not an
        error."""
        conversation_id = await self._convo.get_or_create(channel)
        messages = await self._convo.recent(conversation_id, limit)
        return [m.to_dict() for m in messages]

    async def list_activity(self, limit: int = 100) -> list[dict[str, Any]]:
        """Most recent activity (tool execution) rows, newest first — used by
        the dashboard's /activity page."""
        stmt = select(Activity).order_by(Activity.id.desc()).limit(limit)
        async with self._db.session() as s:
            rows = (await s.execute(stmt)).scalars().all()
        return [
            {
                "id": r.id,
                "tool": r.tool,
                "args": r.args,
                "status": r.status,
                "actor": r.actor,
                "source": r.source,
                "result_preview": r.result_preview,
                "created_at": r.created_at,
            }
            for r in rows
        ]

    async def list_job_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent job_runs rows, newest first — used by the dashboard's
        /jobs page."""
        stmt = select(JobRun).order_by(JobRun.id.desc()).limit(limit)
        async with self._db.session() as s:
            rows = (await s.execute(stmt)).scalars().all()
        return [
            {
                "id": r.id,
                "job_id": r.job_id,
                "status": r.status,
                "started_at": r.started_at,
                "finished_at": r.finished_at,
                "detail": r.detail,
            }
            for r in rows
        ]

    def list_jobs(self) -> list[dict[str, Any]]:
        """Registered jobs with a best-effort next-fire time — used by the
        dashboard's /jobs page. Delegates to JobScheduler, which can answer
        this whether or not the scheduler has actually been started (only
        the single-process assembly calls start(), not Runtime itself)."""
        return self.scheduler.list_jobs()

    async def stop(self) -> None:
        if self._mcp_mount is not None:
            await self._mcp_mount.close()
        await self._db.dispose()
