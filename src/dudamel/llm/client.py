from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

from dudamel.config import BudgetConfig
from dudamel.db import IN_DB_SCOPE, Database
from dudamel.exceptions import BudgetExceededError, LLMError
from dudamel.llm.provider import Provider, ToolSpec
from dudamel.llm.types import Completion, Message
from dudamel.models_core import LlmCall

logger = logging.getLogger("dudamel.llm")

UTC_ZONE = ZoneInfo("UTC")


@dataclass
class Tier:
    name: str
    provider: Provider
    model: str
    max_tokens: int


class LLMClient:
    """The only code that talks to a model."""

    def __init__(
        self,
        tiers: dict[str, Tier],
        db: Database,
        budget: BudgetConfig,
        timezone: ZoneInfo = UTC_ZONE,
    ) -> None:
        self._tiers = tiers
        self._db = db
        self._budget = budget
        self._timezone = timezone

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
        try:
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
        except OperationalError as e:
            # The model call already completed successfully; a DB hiccup
            # recording its usage (e.g. "database is locked") must not turn a
            # good completion into a failure for the caller.
            logger.warning("failed to record llm_calls usage row for tier %s: %s", t.name, e)
        return completion

    async def prompt(self, text: str, *, tier: str = "standard") -> str:
        completion = await self.complete([Message(role="user", text=text)], tier=tier)
        return completion.message.text

    async def _check_budget(self) -> None:
        limit = self._budget.daily_tokens
        if limit is None:
            return
        # The framework's day, not UTC's. This is the one boundary an operator
        # feels directly -- it is the spend cap -- so it follows the same zone
        # as the scheduler and every app rather than being the one exception.
        #
        # `fold=0` is deliberate: in the zones where local midnight does not
        # exist at all (Santiago, Havana, Cairo all skip it on some spring
        # date) it yields the real start of the local day. `fold=1` would put
        # the window an hour early.
        local_midnight = datetime.combine(
            datetime.now(self._timezone).date(), time.min, tzinfo=self._timezone
        )
        # `LlmCall.created_at` is stored naive UTC, so the comparison value has
        # to come back the same way or SQLite compares two different clocks.
        midnight = local_midnight.astimezone(UTC).replace(tzinfo=None)
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
                "raise llm.budget.daily_tokens or wait for the day to roll over "
                "in your configured timezone"
            )
