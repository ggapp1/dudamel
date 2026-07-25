"""Composition root. Interfaces (Plan 3) call chat()/resolve_confirmation()
and nothing else. Construction is in-memory only; start() touches the world."""

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
from dudamel.migrate import upgrade_apps, upgrade_core
from dudamel.models_core import Conversation, PendingConfirmation
from dudamel.orchestrator import Orchestrator
from dudamel.router import ChatReply, Router
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
        """Rebind every app's app.notify() to fn — used by the assembly
        (Plan 3 Task 6) once an interface (e.g. Telegram) is up, replacing the
        WARN-log fallback bound at construction time."""
        for app in self._registry.apps.values():
            app._notify = fn

    async def render_widgets(self) -> list[dict[str, Any]]:
        """Run every registered widget concurrently. Data-plane guarantee: no
        model is ever invoked here (widgets.run_widget calls only widget.fn())."""
        return list(await asyncio.gather(*(run_widget(w) for w in self._registry.widgets)))

    async def stop(self) -> None:
        await self._db.dispose()
