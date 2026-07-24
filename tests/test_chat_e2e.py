# tests/test_chat_e2e.py
"""Plan 2 finish line: full chat flows through Runtime with a scripted
provider — the exact wiring Plan 3's Telegram/web interfaces will call."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from dudamel import App, Orchestrator, Runtime
from dudamel.config import BudgetConfig, Settings, TierConfig
from dudamel.llm.testing import FakeProvider, fake_text, fake_tool_call
from dudamel.migrate import sync_url
from dudamel.models_core import Activity, LlmCall

LOGGED: list[str] = []
WIPED: list[str] = []


def make_orc() -> Orchestrator:
    app = App("gym", description="Workout logging")

    @app.tool
    async def log_workout(exercise: str, reps: int, weight_kg: float) -> str:
        """Record one exercise from today's session."""
        LOGGED.append(f"{exercise}:{reps}:{weight_kg}")
        return f"Logged: {exercise} x{reps} @ {weight_kg}kg"

    @app.tool(confirm=True)
    async def wipe_history(confirm_phrase: str) -> str:
        """Delete all workout history."""
        WIPED.append(confirm_phrase)
        return "history wiped"

    return Orchestrator(apps=[app])


def make_runtime(
    tmp_path: Path, script, budget: BudgetConfig | None = None
) -> tuple[Runtime, FakeProvider, str]:
    url = f"sqlite+aiosqlite:///{tmp_path}/e2e.db"
    settings = Settings(
        database_url=url,
        data_dir=tmp_path,
        llm_tiers={"standard": TierConfig(provider="fake", model="fake-1")},
        llm_budget=budget or BudgetConfig(),
    )
    fp = FakeProvider(script)
    return Runtime(make_orc(), settings, providers={"standard": fp}), fp, url


@pytest.fixture(autouse=True)
def _clear():
    LOGGED.clear()
    WIPED.clear()


async def test_hero_flow_string_args_to_typed_execution(tmp_path) -> None:
    script = [
        fake_tool_call("log_workout", {"exercise": "bench", "reps": "5", "weight_kg": "100"}),
        fake_text("Logged bench 5x100kg — nice session!"),
    ]
    rt, fp, url = make_runtime(tmp_path, script)
    await rt.start()
    reply = await rt.chat("telegram:42", "log bench 5 reps at 100kg", user_id="42")
    assert reply.text == "Logged bench 5x100kg — nice session!"
    assert LOGGED == ["bench:5:100.0"]  # coerced to typed Python
    # bookkeeping: llm usage + activity rows landed
    from sqlalchemy import create_engine

    with create_engine(sync_url(url)).connect() as conn:
        llm_rows = conn.execute(select(LlmCall)).all()
        act_rows = conn.execute(select(Activity)).all()
    assert len(llm_rows) == 2 and len(act_rows) == 1
    await rt.stop()


async def test_confirm_approve_flow(tmp_path) -> None:
    script = [
        fake_tool_call("wipe_history", {"confirm_phrase": "yes really"}),
        fake_text("Everything is gone, as requested."),
    ]
    rt, _fp, _url = make_runtime(tmp_path, script)
    await rt.start()
    r1 = await rt.chat("telegram:42", "wipe my history", user_id="42")
    assert r1.pending_confirmation_id and WIPED == []
    r2 = await rt.resolve_confirmation(r1.pending_confirmation_id, approved=True, user_id="42")
    assert r2.text == "Everything is gone, as requested."
    assert WIPED == ["yes really"]
    await rt.stop()


async def test_confirm_deny_flow(tmp_path) -> None:
    script = [
        fake_tool_call("wipe_history", {"confirm_phrase": "x"}),
        fake_text("Understood — cancelled."),
    ]
    rt, _fp, _url = make_runtime(tmp_path, script)
    await rt.start()
    r1 = await rt.chat("telegram:42", "wipe it", user_id="42")
    r2 = await rt.resolve_confirmation(r1.pending_confirmation_id, approved=False, user_id="42")
    assert r2.text == "Understood — cancelled." and WIPED == []
    await rt.stop()


async def test_budget_exhaustion_is_a_polite_reply(tmp_path) -> None:
    rt, _fp, url = make_runtime(
        tmp_path, [fake_text("never reached")], budget=BudgetConfig(daily_tokens=1)
    )
    await rt.start()
    # seed usage over the limit
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    with Session(create_engine(sync_url(url))) as s:
        s.add(
            LlmCall(
                tier="standard",
                provider="fake",
                model="m",
                tokens_in=5,
                tokens_out=5,
                created_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )
        s.commit()
    reply = await rt.chat("telegram:42", "hi", user_id="42")
    assert "budget" in reply.text.lower()
    await rt.stop()


async def test_cross_surface_continuity(tmp_path) -> None:
    """web chat and telegram share one store — different channels are
    different conversations, same channel resumes."""
    script = [fake_text("first"), fake_text("second")]
    rt, fp, _url = make_runtime(tmp_path, script)
    await rt.start()
    await rt.chat("web:sess1", "hello", user_id="42")
    await rt.chat("web:sess1", "again", user_id="42")
    # second call's window contains the first exchange
    texts = [m.text for m in fp.calls[1]["messages"]]
    assert any("hello" in t for t in texts) and any("first" in t for t in texts)
    await rt.stop()
