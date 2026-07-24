from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from dudamel.config import BudgetConfig
from dudamel.db import Database
from dudamel.exceptions import BudgetExceededError, LLMError
from dudamel.llm.client import LLMClient, Tier
from dudamel.llm.testing import FakeProvider, fake_text
from dudamel.llm.types import Message
from dudamel.migrate import upgrade_core
from dudamel.models_core import LlmCall


@pytest.fixture
async def db(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/c.db"
    upgrade_core(url)
    d = Database(url)
    yield d
    await d.dispose()


def make_client(db: Database, script, budget: BudgetConfig | None = None) -> LLMClient:
    fp = FakeProvider(script)
    tiers = {"standard": Tier(name="standard", provider=fp, model="fake-1", max_tokens=256)}
    return LLMClient(tiers=tiers, db=db, budget=budget or BudgetConfig())


async def test_complete_records_llm_call(db: Database) -> None:
    client = make_client(db, [fake_text("hi", tokens_in=11, tokens_out=7)])
    c = await client.complete([Message(role="user", text="x")])
    assert c.message.text == "hi"
    async with db.session() as s:
        row = (await s.execute(select(LlmCall))).scalar_one()
    assert (row.tier, row.provider, row.model) == ("standard", "fake", "fake-1")
    assert (row.tokens_in, row.tokens_out) == (11, 7)


async def test_unknown_tier(db: Database) -> None:
    client = make_client(db, [])
    with pytest.raises(LLMError, match="standard"):
        await client.complete([Message(role="user", text="x")], tier="nope")


async def test_budget_hard_stop(db: Database) -> None:
    # seed today's usage over the limit, then verify the provider is never called
    async with db.session() as s:
        s.add(
            LlmCall(
                tier="standard",
                provider="fake",
                model="m",
                tokens_in=900,
                tokens_out=200,
                created_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )
    client = make_client(
        db, [fake_text("should never run")], budget=BudgetConfig(daily_tokens=1000)
    )
    with pytest.raises(BudgetExceededError, match="1000"):
        await client.complete([Message(role="user", text="x")])
    fp = client._tiers["standard"].provider
    assert fp.calls == []  # hard stop happened before any provider call


async def test_budget_under_limit_allows(db: Database) -> None:
    client = make_client(db, [fake_text("ok")], budget=BudgetConfig(daily_tokens=10_000))
    c = await client.complete([Message(role="user", text="x")])
    assert c.message.text == "ok"


async def test_warns_inside_db_scope(db: Database, caplog) -> None:
    client = make_client(db, [fake_text("ok")])
    async with db.session():
        await client.complete([Message(role="user", text="x")])
    assert any("inside app.db()" in r.message for r in caplog.records)


async def test_prompt_sugar(db: Database) -> None:
    client = make_client(db, [fake_text("answer")])
    assert await client.prompt("q") == "answer"
