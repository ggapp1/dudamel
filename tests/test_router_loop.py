import asyncio

import pytest
from sqlalchemy import select

from dudamel import App
from dudamel.config import BudgetConfig, RouterConfig
from dudamel.convo import ConversationStore
from dudamel.db import Database
from dudamel.exceptions import LLMError, RegistryError
from dudamel.llm.client import LLMClient, Tier
from dudamel.llm.testing import FakeProvider, fake_text, fake_tool_call
from dudamel.llm.types import Completion, Message, ToolCall, Usage
from dudamel.migrate import upgrade_core
from dudamel.models_core import Activity
from dudamel.registry import Registry
from dudamel.router import ChatReply, Router

CALLS: list[str] = []


def make_app() -> App:
    app = App("gym", description="Workout logging")

    @app.tool
    async def log_workout(exercise: str, reps: int) -> str:
        """Record one exercise."""
        CALLS.append(f"log:{exercise}:{reps}")
        return f"logged {exercise} x{reps}"

    @app.tool(read_only=True, timeout=0.05)
    async def slow_read() -> str:
        """A read that never finishes in time."""
        await asyncio.sleep(1)
        return "late"

    @app.tool(read_only=True)
    async def serialized_probe() -> str:
        """Records interleaving."""
        CALLS.append("probe:start")
        await asyncio.sleep(0.05)
        CALLS.append("probe:end")
        return "ok"

    return app


def make_router(tmp_path, script, config: RouterConfig | None = None):
    url = f"sqlite+aiosqlite:///{tmp_path}/r.db"
    upgrade_core(url)
    db = Database(url)
    fp = FakeProvider(script)
    llm = LLMClient(
        tiers={"standard": Tier(name="standard", provider=fp, model="f", max_tokens=64)},
        db=db,
        budget=BudgetConfig(),
    )
    registry = Registry([make_app()])
    router = Router(
        llm=llm,
        registry=registry,
        convo=ConversationStore(db),
        db=db,
        config=config or RouterConfig(),
    )
    return router, fp, db


@pytest.fixture(autouse=True)
def _clear_calls():
    CALLS.clear()


async def test_plain_text_reply(tmp_path) -> None:
    router, fp, db = make_router(tmp_path, [fake_text("just chatting")])
    reply = await router.handle(channel="t:1", text="hi", user_id="u1")
    assert reply == ChatReply(text="just chatting")
    # system message present, user message delivered
    assert fp.calls[0]["messages"][0].role == "system"
    assert fp.calls[0]["messages"][-1].text == "hi"
    await db.dispose()


async def test_tool_call_roundtrip_with_string_coercion(tmp_path) -> None:
    script = [fake_tool_call("log_workout", {"exercise": "bench", "reps": "5"}), fake_text("Done!")]
    router, fp, db = make_router(tmp_path, script)
    reply = await router.handle(channel="t:1", text="log it", user_id="u1")
    assert reply.text == "Done!" and CALLS == ["log:bench:5"]
    # second model call saw the tool result
    tool_msgs = [m for m in fp.calls[1]["messages"] if m.role == "tool"]
    assert tool_msgs and "logged bench x5" in tool_msgs[0].text
    async with db.session() as s:
        act = (await s.execute(select(Activity))).scalars().all()
    assert [a.status for a in act] == ["ok"]
    await db.dispose()


async def test_unknown_tool_fed_back(tmp_path) -> None:
    script = [fake_tool_call("ghost_tool", {}), fake_text("sorry")]
    router, fp, db = make_router(tmp_path, script)
    reply = await router.handle(channel="t:1", text="x", user_id="u1")
    assert reply.text == "sorry"
    err = [m for m in fp.calls[1]["messages"] if m.role == "tool"][0]
    assert err.is_error and "unknown tool" in err.text and "log_workout" in err.text
    await db.dispose()


async def test_invalid_args_fed_back(tmp_path) -> None:
    script = [
        fake_tool_call("log_workout", {"exercise": "b", "reps": "many"}),
        fake_text("retrying not needed"),
    ]
    router, fp, db = make_router(tmp_path, script)
    await router.handle(channel="t:1", text="x", user_id="u1")
    err = [m for m in fp.calls[1]["messages"] if m.role == "tool"][0]
    assert err.is_error and "invalid arguments" in err.text
    assert CALLS == []  # tool never executed
    await db.dispose()


async def test_tool_timeout_becomes_error_result(tmp_path) -> None:
    script = [fake_tool_call("slow_read", {}), fake_text("ok")]
    router, fp, db = make_router(tmp_path, script)
    await router.handle(channel="t:1", text="x", user_id="u1")
    err = [m for m in fp.calls[1]["messages"] if m.role == "tool"][0]
    assert err.is_error and "timed out" in err.text
    await db.dispose()


async def test_parallel_tool_calls_one_turn(tmp_path) -> None:
    both = Completion(
        message=Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="a", name="log_workout", args={"exercise": "squat", "reps": 3}),
                ToolCall(id="b", name="log_workout", args={"exercise": "bench", "reps": 5}),
            ],
        ),
        usage=Usage(1, 1),
        stop_reason="tool_calls",
    )
    router, fp, db = make_router(tmp_path, [both, fake_text("both done")])
    reply = await router.handle(channel="t:1", text="x", user_id="u1")
    assert reply.text == "both done" and sorted(CALLS) == ["log:bench:5", "log:squat:3"]
    results = [m for m in fp.calls[1]["messages"] if m.role == "tool"]
    assert [r.tool_call_id for r in results] == ["a", "b"]  # order preserved
    await db.dispose()


async def test_duplicate_tool_call_ids_both_execute(tmp_path) -> None:
    both = Completion(
        message=Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="dup", name="log_workout", args={"exercise": "a", "reps": 1}),
                ToolCall(id="dup", name="log_workout", args={"exercise": "b", "reps": 2}),
            ],
        ),
        usage=Usage(1, 1),
        stop_reason="tool_calls",
    )
    router, fp, db = make_router(tmp_path, [both, fake_text("done")])
    reply = await router.handle(channel="t:1", text="x", user_id="u1")
    assert reply.text == "done" and sorted(CALLS) == ["log:a:1", "log:b:2"]
    results = [m for m in fp.calls[1]["messages"] if m.role == "tool"]
    assert len(results) == 2
    texts = {r.text for r in results}
    assert any("a x1" in t for t in texts)
    assert any("b x2" in t for t in texts)
    await db.dispose()


async def test_iteration_cap(tmp_path) -> None:
    script = [
        fake_tool_call("log_workout", {"exercise": "e", "reps": 1}, id=f"i{n}") for n in range(9)
    ]
    router, fp, db = make_router(tmp_path, script, RouterConfig(iteration_cap=2))
    reply = await router.handle(channel="t:1", text="x", user_id="u1")
    assert "couldn't finish" in reply.text and len(fp.calls) == 2
    await db.dispose()


async def test_mid_loop_model_death_after_side_effect(tmp_path) -> None:
    script = [
        fake_tool_call("log_workout", {"exercise": "e", "reps": 1}),
        LLMError("connection reset", retryable=True),
    ]
    router, fp, db = make_router(tmp_path, script)
    reply = await router.handle(channel="t:1", text="x", user_id="u1")
    assert "completed the action" in reply.text and CALLS == ["log:e:1"]
    await db.dispose()


async def test_model_down_before_any_action(tmp_path) -> None:
    router, fp, db = make_router(tmp_path, [LLMError("refused")])
    reply = await router.handle(channel="t:1", text="x", user_id="u1")
    assert "unavailable" in reply.text
    await db.dispose()


async def test_duplicate_client_msg_id(tmp_path) -> None:
    script = [fake_text("first")]
    router, fp, db = make_router(tmp_path, script)
    r1 = await router.handle(channel="t:1", text="x", user_id="u1", client_msg_id="m1")
    r2 = await router.handle(channel="t:1", text="x", user_id="u1", client_msg_id="m1")
    assert r1.text == "first" and r2 == ChatReply(text="")
    assert len(fp.calls) == 1  # no second model call
    await db.dispose()


async def test_per_conversation_serialization(tmp_path) -> None:
    script = [
        fake_tool_call("serialized_probe", {}),
        fake_text("one"),
        fake_tool_call("serialized_probe", {}),
        fake_text("two"),
    ]
    router, fp, db = make_router(tmp_path, script)
    await asyncio.gather(
        router.handle(channel="t:1", text="a", user_id="u1"),
        router.handle(channel="t:1", text="b", user_id="u1"),
    )
    assert CALLS == ["probe:start", "probe:end", "probe:start", "probe:end"]
    await db.dispose()


async def test_tool_results_capped_at_rest(tmp_path) -> None:
    """I4: tool results are capped not just in the window sent to the model,
    but at rest in the conversation store."""
    app = App("gym", description="d")

    @app.tool
    async def huge_result() -> str:
        """Return a huge result."""
        return "B" * 50_000

    url = f"sqlite+aiosqlite:///{tmp_path}/cap.db"
    upgrade_core(url)
    db = Database(url)
    script = [fake_tool_call("huge_result", {}), fake_text("done")]
    fp = FakeProvider(script)
    llm = LLMClient(
        tiers={"standard": Tier(name="standard", provider=fp, model="f", max_tokens=64)},
        db=db,
        budget=BudgetConfig(),
    )
    registry = Registry([app])
    convo = ConversationStore(db)
    router = Router(
        llm=llm,
        registry=registry,
        convo=convo,
        db=db,
        config=RouterConfig(tool_result_cap=1000),
    )
    reply = await router.handle(channel="t:1", text="go", user_id="u1")
    assert reply.text == "done"
    cid = await convo.get_or_create("t:1")
    history = await convo.recent(cid)
    tool_msg = [m for m in history if m.role == "tool"][0]
    assert len(tool_msg.text) < 1100
    assert "truncated" in tool_msg.text
    await db.dispose()


async def test_tool_ceiling_enforced_at_construction(tmp_path) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path}/c.db"
    upgrade_core(url)
    db = Database(url)
    app = App("many", description="d")
    for i in range(3):

        async def fn() -> str:
            """Doc."""
            return "x"

        fn.__name__ = f"tool_{i}"
        app._register_tool(fn, read_only=True, confirm=False, timeout=30.0)
    llm = LLMClient(tiers={}, db=db, budget=BudgetConfig())
    with pytest.raises(RegistryError, match="max_tools"):
        Router(
            llm=llm,
            registry=Registry([app]),
            convo=ConversationStore(db),
            db=db,
            config=RouterConfig(max_tools=2),
        )
    await db.dispose()
