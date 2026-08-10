"""Composition root. Interfaces call chat()/resolve_confirmation() and
nothing else. Construction is in-memory only; start() touches the world."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy import select

from dudamel.config import Settings
from dudamel.convo import ConversationStore
from dudamel.db import Database
from dudamel.exceptions import LLMError, RegistryError
from dudamel.llm.anthropic import AnthropicProvider
from dudamel.llm.client import LLMClient, Tier
from dudamel.llm.openai_compat import OpenAICompatProvider
from dudamel.llm.provider import Provider
from dudamel.llm.types import Message
from dudamel.mcp_mount import MCPMount
from dudamel.migrate import upgrade_apps, upgrade_core
from dudamel.models_core import Activity, Conversation, JobRun, PendingConfirmation
from dudamel.orchestrator import Orchestrator
from dudamel.router import ChatReply, Router
from dudamel.scheduler import JobScheduler
from dudamel.widgets import run_widget

logger = logging.getLogger("dudamel.runtime")


class Runtime:
    def __init__(
        self,
        orchestrator: Orchestrator,
        settings: Settings,
        *,
        providers: dict[str, Provider] | None = None,
    ) -> None:
        self._settings = settings
        self._db = Database(settings.database_url)
        self._registry = orchestrator.registry
        self._mcp_commands = list(orchestrator.mcp)
        self._mcp_mount: MCPMount | None = None
        tiers = self._build_tiers(providers or {})
        self._llm = LLMClient(tiers=tiers, db=self._db, budget=settings.llm_budget)
        self._convo = ConversationStore(self._db)
        self._router = Router(
            llm=self._llm,
            registry=self._registry,
            convo=self._convo,
            db=self._db,
            config=settings.router,
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
            elif cfg.provider == "openai-compatible":
                if not cfg.base_url:
                    raise RegistryError(f"tier {name!r}: openai-compatible provider needs base_url")
                key = os.environ.get(cfg.api_key_env, "unused") if cfg.api_key_env else "unused"
                provider = OpenAICompatProvider(base_url=cfg.base_url, api_key=key)
            elif cfg.provider == "anthropic":
                env = cfg.api_key_env or "ANTHROPIC_API_KEY"
                key = os.environ.get(env)
                if not key:
                    raise RegistryError(f"tier {name!r}: environment variable {env} is not set")
                provider = AnthropicProvider(api_key=key)
            else:  # "fake"
                raise RegistryError(
                    f"tier {name!r}: provider 'fake' requires a providers= override "
                    "(dudamel.llm.testing.FakeProvider)"
                )
            tiers[name] = Tier(
                name=name, provider=provider, model=cfg.model, max_tokens=cfg.max_tokens
            )
        for name in overrides:
            if name not in tiers:  # override for a tier absent from config
                cfg = self._settings.llm_tiers.get(name)
                model = cfg.model if cfg else "override"
                tiers[name] = Tier(
                    name=name, provider=overrides[name], model=model, max_tokens=1024
                )
        return tiers

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
        await asyncio.to_thread(upgrade_core, url)
        migrations_dir = self._settings.data_dir / "migrations"
        if migrations_dir.exists():
            await asyncio.to_thread(upgrade_apps, url, self._settings.data_dir)
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

    async def list_pending_confirmations(self, channel: str | None = None) -> list[dict[str, Any]]:
        stmt = select(PendingConfirmation).where(PendingConfirmation.status == "pending")
        if channel is not None:
            stmt = stmt.join(
                Conversation, Conversation.id == PendingConfirmation.conversation_id
            ).where(Conversation.channel == channel)
        stmt = stmt.order_by(PendingConfirmation.created_at)
        async with self._db.session() as s:
            rows = (await s.execute(stmt)).scalars().all()
        return [
            {
                "id": r.id,
                "tool": r.tool,
                "args": r.args,
                "created_at": r.created_at,
                "expires_at": r.expires_at,
            }
            for r in rows
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

    async def render_widgets(self) -> list[dict[str, Any]]:
        """Run every registered widget concurrently. Data-plane guarantee: no
        model is ever invoked here (widgets.run_widget calls only widget.fn())."""
        return list(await asyncio.gather(*(run_widget(w) for w in self._registry.widgets)))

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
