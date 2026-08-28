from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from dudamel.config import BudgetConfig
from dudamel.db import Database
from dudamel.exceptions import BudgetExceededError, LLMError
from dudamel.llm.client import UTC_ZONE, LLMClient, Tier
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


def make_client(
    db: Database,
    script,
    budget: BudgetConfig | None = None,
    timezone: ZoneInfo | None = None,
) -> LLMClient:
    fp = FakeProvider(script)
    tiers = {"standard": Tier(name="standard", provider=fp, model="fake-1", max_tokens=256)}
    return LLMClient(
        tiers=tiers, db=db, budget=budget or BudgetConfig(), timezone=timezone or UTC_ZONE
    )


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


# --- Rider B: usage-insert must not fail an otherwise-completed call --------


async def test_usage_insert_operational_error_does_not_fail_completion(
    db: Database, caplog, monkeypatch
) -> None:
    client = make_client(db, [fake_text("hi")])

    def _raising_factory(*args: object, **kwargs: object) -> None:
        raise OperationalError("insert into llm_calls", {}, Exception("database is locked"))

    # Monkeypatch the session factory so the usage-row insert -- the only DB
    # write complete() does when no budget is configured -- fails.
    monkeypatch.setattr(db, "_factory", _raising_factory)

    c = await client.complete([Message(role="user", text="x")])

    assert c.message.text == "hi"  # the completed call still succeeds
    assert any("llm_calls" in r.message for r in caplog.records)
    assert any(r.levelname == "WARNING" for r in caplog.records)


# --- the spend cap resets on the framework's day, not UTC's -----------------


def _freeze_client_clock(monkeypatch, instant: datetime) -> None:
    from dudamel.llm import client as client_mod

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    monkeypatch.setattr(client_mod, "datetime", _Frozen)


async def test_the_budget_day_rolls_over_in_the_framework_zone(db: Database, monkeypatch) -> None:
    """The framework has one day boundary, and the spend cap is not allowed to
    be the exception -- an operator whose day starts at 00:00 local should not
    find their budget reset at 21:00.

    04:59Z on 2026-01-16 is still 2026-01-15 in New York, so this spend sits
    outside today's local window. The UTC boundary counted it.
    """
    async with db.session() as s:
        s.add(
            LlmCall(
                tier="standard",
                provider="fake",
                model="m",
                tokens_in=900,
                tokens_out=200,
                created_at=datetime(2026, 1, 16, 4, 59),
            )
        )
    _freeze_client_clock(monkeypatch, datetime(2026, 1, 16, 14, 0, tzinfo=UTC))
    client = make_client(
        db,
        [fake_text("ok")],
        budget=BudgetConfig(daily_tokens=1000),
        timezone=ZoneInfo("America/New_York"),
    )
    assert (await client.complete([Message(role="user", text="x")])).message.text == "ok"


async def test_spend_inside_the_local_day_still_exhausts_the_budget(
    db: Database, monkeypatch
) -> None:
    """The other half of the boundary. 06:00Z on 2026-01-16 IS 2026-01-16 in
    New York, so it must count -- without this row, an implementation that
    simply counts nothing satisfies the test above.
    """
    async with db.session() as s:
        s.add(
            LlmCall(
                tier="standard",
                provider="fake",
                model="m",
                tokens_in=900,
                tokens_out=200,
                created_at=datetime(2026, 1, 16, 6, 0),
            )
        )
    _freeze_client_clock(monkeypatch, datetime(2026, 1, 16, 14, 0, tzinfo=UTC))
    client = make_client(
        db,
        [fake_text("should never run")],
        budget=BudgetConfig(daily_tokens=1000),
        timezone=ZoneInfo("America/New_York"),
    )
    with pytest.raises(BudgetExceededError, match="1000"):
        await client.complete([Message(role="user", text="x")])


async def test_the_exhausted_message_points_at_the_configured_zone(
    db: Database, monkeypatch
) -> None:
    """The message used to promise "the UTC day", which is no longer the
    boundary it describes."""
    async with db.session() as s:
        s.add(
            LlmCall(
                tier="standard",
                provider="fake",
                model="m",
                tokens_in=900,
                tokens_out=200,
                created_at=datetime(2026, 1, 16, 6, 0),
            )
        )
    _freeze_client_clock(monkeypatch, datetime(2026, 1, 16, 14, 0, tzinfo=UTC))
    client = make_client(
        db,
        [fake_text("should never run")],
        budget=BudgetConfig(daily_tokens=1000),
        timezone=ZoneInfo("America/New_York"),
    )
    with pytest.raises(BudgetExceededError) as excinfo:
        await client.complete([Message(role="user", text="x")])
    assert "UTC day" not in str(excinfo.value)
    assert "configured timezone" in str(excinfo.value)
