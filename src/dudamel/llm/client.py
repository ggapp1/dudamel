from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, time
from typing import Any

from sqlalchemy import func, select

from dudamel.config import BudgetConfig
from dudamel.db import IN_DB_SCOPE, Database
from dudamel.exceptions import BudgetExceededError, LLMError
from dudamel.llm.provider import Provider, ToolSpec
from dudamel.llm.types import Completion, Message
from dudamel.models_core import LlmCall

logger = logging.getLogger("dudamel.llm")


@dataclass
class Tier:
    name: str
    provider: Provider
    model: str
    max_tokens: int


class LLMClient:
    """The only code that talks to a model."""

    def __init__(self, tiers: dict[str, Tier], db: Database, budget: BudgetConfig) -> None:
        self._tiers = tiers
        self._db = db
        self._budget = budget
        if budget.daily_usd is not None:
            logger.warning(
                "llm.budget.daily_usd is configured but v1 enforces token budgets "
                "only — set daily_tokens for a hard ceiling"
            )

    async def complete(
        self,
        messages: list[Message],
        *,
        tier: str = "standard",
        tools: list[ToolSpec] | None = None,
        json_schema: dict[str, Any] | None = None,
        conversation_id: int | None = None,
    ) -> Completion:
        if IN_DB_SCOPE.get():
            logger.warning(
                "model call issued inside app.db() scope — never hold a DB "
                "transaction across an LLM request (SQLite writer lock)"
            )
        t = self._tiers.get(tier)
        if t is None:
            raise LLMError(
                f"unknown tier {tier!r}; configured tiers: {sorted(self._tiers) or 'none'}"
            )
        await self._check_budget()
        completion = await t.provider.complete(
            model=t.model,
            messages=messages,
            tools=tools,
            max_tokens=t.max_tokens,
            json_schema=json_schema,
        )
        async with self._db.session() as s:
            s.add(
                LlmCall(
                    tier=t.name,
                    provider=t.provider.name,
                    model=t.model,
                    tokens_in=completion.usage.tokens_in,
                    tokens_out=completion.usage.tokens_out,
                    conversation_id=conversation_id,
                )
            )
        return completion

    async def prompt(self, text: str, *, tier: str = "standard") -> str:
        completion = await self.complete([Message(role="user", text=text)], tier=tier)
        return completion.message.text

    async def _check_budget(self) -> None:
        limit = self._budget.daily_tokens
        if limit is None:
            return
        midnight = datetime.combine(datetime.now(UTC).date(), time.min)
        async with self._db.session() as s:
            spent = (
                await s.execute(
                    select(
                        func.coalesce(func.sum(LlmCall.tokens_in + LlmCall.tokens_out), 0)
                    ).where(LlmCall.created_at >= midnight)
                )
            ).scalar_one()
        if spent >= limit:
            raise BudgetExceededError(
                f"daily token budget exhausted ({spent}/{limit}); "
                "raise llm.budget.daily_tokens or wait for the UTC day to roll over"
            )
