import asyncio
import logging

import pytest
from sqlalchemy import select

from dudamel import App
from dudamel.compaction import Compactor
from dudamel.config import BudgetConfig, RouterConfig
from dudamel.contract.schema import ToolSchema
from dudamel.contract.types import Tool
from dudamel.convo import ConversationStore
from dudamel.db import Database
from dudamel.exceptions import LLMError, RegistryError
from dudamel.llm.client import LLMClient, Tier
from dudamel.llm.testing import FakeProvider, fake_text, fake_tool_call
from dudamel.llm.types import Completion, Message, ToolCall, Usage
from dudamel.migrate import upgrade_core
from dudamel.models_core import Activity, Summary
from dudamel.registry import Registry
from dudamel.router import ChatReply, Router, select_tool_subset
from dudamel.window import message_tokens

CALLS: list[str] = []


def _mcp_tool(name: str, description: str, *, origin: str = "mcp") -> Tool:
    """A minimal zero-argument tool for subsetting tests -- shape only,
    the fn is never meant to be called."""

    async def fn() -> str:
        return "ok"

    fn.__name__ = name
    fn.__doc__ = description
    return Tool(
        name=name,
        app_name="ext",
        description=description,
        fn=fn,
        schema=ToolSchema(fn),
        read_only=True,
        confirm=False,
        timeout=30.0,
        origin=origin,
    )


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


def test_persona_replaces_identity_line_but_never_the_apps_or_tool_instruction(tmp_path) -> None:
    """The persona swaps only who the assistant says it is; the installed-apps
    block and the tool-use instruction are structural and always present, so a
    persona cannot disable tool use by accident."""
    config = RouterConfig(iteration_cap=8)
    config.persona = "You are Jeeves, a butler."
    router, fp, db = make_router(tmp_path, [fake_text("ok")], config=config)
    msg = router._system_message()
    assert "You are Jeeves, a butler." in msg.text
    assert "Installed apps:" in msg.text
    assert "gym:" in msg.text
    assert "Use the available tools" in msg.text
    assert "You are dudamel" not in msg.text


def test_persona_default_keeps_the_original_identity_line(tmp_path) -> None:
    """When persona is None or not set, the original identity line is used."""
    router, fp, db = make_router(tmp_path, [fake_text("ok")], config=RouterConfig())
    msg = router._system_message()
    assert "You are dudamel, a personal assistant orchestrator." in msg.text
    assert "Installed apps:" in msg.text
    assert "Use the available tools" in msg.text


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


# -- per-turn tool subsetting past the mcp overflow ceiling ------------------
# refresh_tool_specs() used to permanently DELETE excess mcp tools from the
# registry once mounting pushed past max_tools. It no longer does: every
# tool stays registered, and each turn is offered a relevant subset,
# selected by select_tool_subset().


def test_native_tools_are_never_dropped_by_subsetting() -> None:
    tools = {
        "native_a": _mcp_tool("native_a", "totally unrelated to the query", origin="native"),
        "native_b": _mcp_tool("native_b", "also unrelated to anything", origin="native"),
        "mcp_x": _mcp_tool("mcp_x", "matches query word banana"),
        "mcp_y": _mcp_tool("mcp_y", "matches query word banana too"),
    }
    # max_tools=1 is smaller than the 2 native tools alone -- both natives
    # must still survive, even though that overruns the nominal ceiling.
    selected = select_tool_subset(tools, max_tools=1, query="banana", must_keep=set())
    assert set(selected) == {"native_a", "native_b"}


def test_select_tool_subset_ranks_by_overlap_then_name() -> None:
    """Overlap decides who gets a slot; the name is only the tie-break.

    Hand-computed against the query tokens {citrus, harvest, schedule},
    scoring each tool's own name + description:

        mcp_gamma  citrus harvest schedule planner  -> 3
        mcp_beta   citrus only                      -> 1
        mcp_delta  citrus mentioned once            -> 1
        mcp_alpha  zephyr unrelated content         -> 0

    Ranked, that is gamma, beta, delta, alpha -- beta ahead of delta purely
    on name, gamma ahead of both on score. The alphabetical order is a
    different one (alpha, beta, delta, gamma), so taking the top three by
    rank keeps gamma and drops alpha, while taking the top three by name
    would do the exact opposite. The two orders cannot be confused here.
    """
    tools = {
        "mcp_alpha": _mcp_tool("mcp_alpha", "zephyr unrelated content"),
        "mcp_gamma": _mcp_tool("mcp_gamma", "citrus harvest schedule planner"),
        "mcp_beta": _mcp_tool("mcp_beta", "citrus only"),
        "mcp_delta": _mcp_tool("mcp_delta", "citrus mentioned once"),
    }
    selected = select_tool_subset(
        tools, max_tools=3, query="citrus harvest schedule", must_keep=set()
    )
    # The return value is name-sorted for stability, so this asserts WHICH
    # three the ranking chose, not the ranking's own order.
    assert selected == ["mcp_beta", "mcp_delta", "mcp_gamma"]
    # Two slots would cut at the tie: gamma on score, then beta on name.
    assert select_tool_subset(
        tools, max_tools=2, query="citrus harvest schedule", must_keep=set()
    ) == ["mcp_beta", "mcp_gamma"]


async def test_mcp_overflow_subsets_per_turn_instead_of_deleting(tmp_path, caplog) -> None:
    """Mounting past max_tools must not permanently discard any mcp tool: a
    later turn about a different topic gets a different subset, and nothing
    the first turn left out is gone from the registry."""
    app = App("gym", description="d")

    @app.tool
    async def log_workout(exercise: str) -> str:
        """Record a workout."""
        return "ok"

    @app.tool(read_only=True)
    async def status() -> str:
        """Report status."""
        return "ok"

    registry = Registry([app])  # 2 native tools -- fits under max_tools=4 at construction

    url = f"sqlite+aiosqlite:///{tmp_path}/overflow.db"
    upgrade_core(url)
    db = Database(url)
    fp = FakeProvider([fake_text("weather is sunny"), fake_text("stocks are up")])
    llm = LLMClient(
        tiers={"standard": Tier(name="standard", provider=fp, model="f", max_tokens=64)},
        db=db,
        budget=BudgetConfig(),
    )
    convo = ConversationStore(db)
    router = Router(
        llm=llm, registry=registry, convo=convo, db=db, config=RouterConfig(max_tools=4)
    )
    # Mirrors Runtime.start(): mcp mounting adds tools to the registry after
    # the Router already exists, pushing this batch past max_tools.
    registry.tools["weather_forecast"] = _mcp_tool(
        "weather_forecast", "Get the weather forecast for a city"
    )
    registry.tools["weather_alerts"] = _mcp_tool(
        "weather_alerts", "Get weather alerts and warnings"
    )
    registry.tools["stock_price"] = _mcp_tool(
        "stock_price", "Get the current stock price for a ticker"
    )
    registry.tools["stock_history"] = _mcp_tool("stock_history", "Get historical stock price data")
    router.refresh_tool_specs()

    with caplog.at_level(logging.WARNING, logger="dudamel.router"):
        r1 = await router.handle(channel="t:1", text="weather forecast please", user_id="u1")
    assert r1.text == "weather is sunny"
    offered1 = {s.name for s in fp.calls[0]["tools"]}
    assert offered1 == {"log_workout", "status", "weather_forecast", "weather_alerts"}
    # not deleted -- still in the registry, just not offered this turn
    assert set(registry.tools) == {
        "log_workout",
        "status",
        "weather_forecast",
        "weather_alerts",
        "stock_price",
        "stock_history",
    }
    assert "stock_price" in caplog.text and "stock_history" in caplog.text

    r2 = await router.handle(channel="t:2", text="stock price please", user_id="u1")
    assert r2.text == "stocks are up"
    offered2 = {s.name for s in fp.calls[1]["tools"]}
    assert offered2 == {"log_workout", "status", "stock_price", "stock_history"}
    await db.dispose()


async def test_tools_called_earlier_in_the_turn_stay_offered(tmp_path) -> None:
    """A tool the model already invoked this turn must remain visible on the
    next iteration even if lexical ranking would now exclude it."""
    app = App("gym", description="d")

    @app.tool
    async def log_workout(exercise: str) -> str:
        """Record a workout."""
        return "ok"

    registry = Registry([app])  # 1 native tool -- fits under max_tools=2 at construction

    url = f"sqlite+aiosqlite:///{tmp_path}/keep.db"
    upgrade_core(url)
    db = Database(url)
    # The model calls the low-scoring tool directly -- FakeProvider doesn't
    # consult what was offered, only what real providers would refuse.
    fp = FakeProvider([fake_tool_call("mcp_low", {}), fake_text("done")])
    llm = LLMClient(
        tiers={"standard": Tier(name="standard", provider=fp, model="f", max_tokens=64)},
        db=db,
        budget=BudgetConfig(),
    )
    convo = ConversationStore(db)
    router = Router(
        llm=llm, registry=registry, convo=convo, db=db, config=RouterConfig(max_tools=2)
    )
    registry.tools["mcp_high"] = _mcp_tool("mcp_high", "banana banana banana query match")
    registry.tools["mcp_low"] = _mcp_tool("mcp_low", "completely unrelated content zephyr")
    router.refresh_tool_specs()

    reply = await router.handle(channel="t:1", text="banana banana", user_id="u1")
    assert reply.text == "done"
    offered_iter1 = {s.name for s in fp.calls[0]["tools"]}
    assert offered_iter1 == {"log_workout", "mcp_high"}  # ranked in on relevance
    offered_iter2 = {s.name for s in fp.calls[1]["tools"]}
    # mcp_low was just called -- it must stay visible even though it
    # crowds out the higher-ranked mcp_high on the next iteration.
    assert offered_iter2 == {"log_workout", "mcp_low"}
    await db.dispose()


def _subset_warnings(caplog) -> list[str]:
    """The "left out of this turn" WARN records only -- refresh_tool_specs()
    emits its own, once at mount time, on the same logger."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "not offered this turn" in r.getMessage()
    ]


async def test_subsetting_warns_once_per_turn_not_once_per_iteration(tmp_path, caplog) -> None:
    """The WARN names which tools a turn left out. A turn takes as many
    model calls as it needs; repeating the same notice for each of them
    turns one operator-relevant fact into log noise proportional to how
    tool-heavy the turn happened to be."""
    app = App("gym", description="d")

    @app.tool
    async def log_workout(exercise: str) -> str:
        """Record a workout."""
        return "ok"

    registry = Registry([app])  # 1 native tool -- fits under max_tools=2 at construction

    url = f"sqlite+aiosqlite:///{tmp_path}/warnonce.db"
    upgrade_core(url)
    db = Database(url)
    # tool call -> tool result -> second completion: two subsetting iterations.
    fp = FakeProvider([fake_tool_call("log_workout", {"exercise": "run"}), fake_text("done")])
    llm = LLMClient(
        tiers={"standard": Tier(name="standard", provider=fp, model="f", max_tokens=64)},
        db=db,
        budget=BudgetConfig(),
    )
    router = Router(
        llm=llm,
        registry=registry,
        convo=ConversationStore(db),
        db=db,
        config=RouterConfig(max_tools=2),
    )
    registry.tools["mcp_a"] = _mcp_tool("mcp_a", "banana fetch data alpha")
    registry.tools["mcp_b"] = _mcp_tool("mcp_b", "completely unrelated beta")
    router.refresh_tool_specs()

    with caplog.at_level(logging.WARNING, logger="dudamel.router"):
        reply = await router.handle(channel="t:1", text="banana fetch", user_id="u1")
    assert reply.text == "done"
    assert len(fp.calls) == 2  # the turn really did subset twice
    warnings = _subset_warnings(caplog)
    assert len(warnings) == 1 and "mcp_b" in warnings[0]
    await db.dispose()


async def test_at_the_ceiling_nothing_is_subset_and_nothing_warns(tmp_path, caplog) -> None:
    """`len(tools) == max_tools` is the boundary the overflow check reads as
    "fits": every tool is offered and the turn is silent. Only strictly
    exceeding the ceiling starts leaving tools out."""
    app = App("gym", description="d")

    @app.tool
    async def log_workout(exercise: str) -> str:
        """Record a workout."""
        return "ok"

    @app.tool(read_only=True)
    async def status() -> str:
        """Report status."""
        return "ok"

    registry = Registry([app])  # 2 native tools

    url = f"sqlite+aiosqlite:///{tmp_path}/ceiling.db"
    upgrade_core(url)
    db = Database(url)
    fp = FakeProvider([fake_text("all good")])
    llm = LLMClient(
        tiers={"standard": Tier(name="standard", provider=fp, model="f", max_tokens=64)},
        db=db,
        budget=BudgetConfig(),
    )
    router = Router(
        llm=llm,
        registry=registry,
        convo=ConversationStore(db),
        db=db,
        config=RouterConfig(max_tools=4),
    )
    registry.tools["mcp_a"] = _mcp_tool("mcp_a", "banana fetch data alpha")
    registry.tools["mcp_b"] = _mcp_tool("mcp_b", "completely unrelated beta")
    assert len(registry.tools) == 4  # exactly at max_tools, not past it

    with caplog.at_level(logging.WARNING, logger="dudamel.router"):
        router.refresh_tool_specs()
        reply = await router.handle(channel="t:1", text="banana fetch", user_id="u1")
    assert reply.text == "all good"
    assert {s.name for s in fp.calls[0]["tools"]} == set(registry.tools)
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
    await db.dispose()


async def test_resumed_turn_sees_the_same_subset_it_was_shown(tmp_path) -> None:
    """Confirmation resume rebuilds the spec list from the persisted names,
    not by re-ranking -- re-ranking against a stale user message (or a
    registry that changed in the meantime) could swap the tool set mid-turn."""
    app = App("gym", description="d")

    @app.tool(confirm=True)
    async def wipe_log(reason: str) -> str:
        """Delete the whole workout log."""
        return "wiped"

    registry = Registry([app])  # 1 native tool -- fits under max_tools=2 at construction

    url = f"sqlite+aiosqlite:///{tmp_path}/resume.db"
    upgrade_core(url)
    db = Database(url)
    fp = FakeProvider([fake_tool_call("wipe_log", {"reason": "go"}), fake_text("resumed done")])
    llm = LLMClient(
        tiers={"standard": Tier(name="standard", provider=fp, model="f", max_tokens=64)},
        db=db,
        budget=BudgetConfig(),
    )
    convo = ConversationStore(db)
    router = Router(
        llm=llm, registry=registry, convo=convo, db=db, config=RouterConfig(max_tools=2)
    )
    registry.tools["mcp_a"] = _mcp_tool("mcp_a", "banana fetch data alpha")
    registry.tools["mcp_b"] = _mcp_tool("mcp_b", "completely unrelated beta")
    router.refresh_tool_specs()

    r1 = await router.handle(channel="t:1", text="banana fetch", user_id="u1")
    assert r1.pending_confirmation_id is not None
    offered_iter0 = {s.name for s in fp.calls[0]["tools"]}
    assert offered_iter0 == {"wipe_log", "mcp_a"}

    # Simulate a tool landing between suspension and resume that would win
    # the ranking if the resumed turn re-ranked instead of reusing the
    # persisted subset.
    registry.tools["mcp_new"] = _mcp_tool("mcp_new", "banana fetch supreme match")

    r2 = await router.resolve_confirmation(r1.pending_confirmation_id, approved=True, user_id="u1")
    assert r2.text == "resumed done"
    offered_iter1 = {s.name for s in fp.calls[1]["tools"]}
    assert offered_iter1 == {"wipe_log", "mcp_a"}
    assert "mcp_new" not in offered_iter1
    await db.dispose()


async def test_tools_called_before_a_confirm_resume_stay_offered_after_it(tmp_path) -> None:
    """A confirm gate splits one turn across two calls into the loop. The
    "a tool this turn already used stays visible" rule has to survive that
    split: the model that resumes is mid-task, and losing the tool it was
    working with -- to a newly mounted server that merely ranks higher --
    strands it exactly where it needs a follow-up call.

    Ranking here is hand-computed against the query "banana fetch":
    mcp_used scores 1 (banana), mcp_alt 0, and mcp_new -- which only
    appears during the suspension -- scores 2 (banana, fetch). With three
    slots and two always-retained native tools, one slot is up for grabs,
    so after the resume mcp_new outranks mcp_used for it.
    """
    app = App("gym", description="d")

    @app.tool(confirm=True)
    async def wipe_log(reason: str) -> str:
        """Delete the whole workout log."""
        return "wiped"

    @app.tool(read_only=True)
    async def status() -> str:
        """Report status."""
        return "ok"

    registry = Registry([app])  # 2 native tools -- fits under max_tools=3 at construction

    url = f"sqlite+aiosqlite:///{tmp_path}/reseed.db"
    upgrade_core(url)
    db = Database(url)
    # One batch calling an mcp tool and the confirm-gated native tool, then
    # (after the resume) a native call to force a further subsetting
    # iteration, then the closing text.
    batch = Completion(
        message=Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="tc1", name="mcp_used", args={}),
                ToolCall(id="tc2", name="wipe_log", args={"reason": "go"}),
            ],
        ),
        usage=Usage(tokens_in=10, tokens_out=5),
        stop_reason="tool_calls",
    )
    fp = FakeProvider([batch, fake_tool_call("status", {}), fake_text("all done")])
    llm = LLMClient(
        tiers={"standard": Tier(name="standard", provider=fp, model="f", max_tokens=64)},
        db=db,
        budget=BudgetConfig(),
    )
    router = Router(
        llm=llm,
        registry=registry,
        convo=ConversationStore(db),
        db=db,
        config=RouterConfig(max_tools=3),
    )
    registry.tools["mcp_used"] = _mcp_tool("mcp_used", "banana zephyr log")
    registry.tools["mcp_alt"] = _mcp_tool("mcp_alt", "quux irrelevant thing")
    router.refresh_tool_specs()

    r1 = await router.handle(channel="t:1", text="banana fetch", user_id="u1")
    assert r1.pending_confirmation_id is not None
    assert {s.name for s in fp.calls[0]["tools"]} == {"wipe_log", "status", "mcp_used"}

    # A server mounts (or reconnects) while the user is deciding, bringing a
    # tool that outranks the one the turn is already working with.
    registry.tools["mcp_new"] = _mcp_tool("mcp_new", "banana fetch supreme")

    r2 = await router.resolve_confirmation(r1.pending_confirmation_id, approved=True, user_id="u1")
    assert r2.text == "all done"
    # Iteration after the resume replays the persisted subset verbatim.
    assert {s.name for s in fp.calls[1]["tools"]} == {"wipe_log", "status", "mcp_used"}
    # The one after that re-ranks live -- and must still carry mcp_used,
    # which this turn called before the confirm gate suspended it.
    assert {s.name for s in fp.calls[2]["tools"]} == {"wipe_log", "status", "mcp_used"}
    await db.dispose()


async def test_vanished_tool_in_persisted_subset_is_skipped_not_fatal(tmp_path) -> None:
    """A server can drop a tool between suspension and resume; the resumed
    turn proceeds with the survivors instead of failing."""
    app = App("gym", description="d")

    @app.tool(confirm=True)
    async def wipe_log(reason: str) -> str:
        """Delete the whole workout log."""
        return "wiped"

    registry = Registry([app])  # 1 native tool -- fits under max_tools=2 at construction

    url = f"sqlite+aiosqlite:///{tmp_path}/vanish.db"
    upgrade_core(url)
    db = Database(url)
    fp = FakeProvider([fake_tool_call("wipe_log", {"reason": "go"}), fake_text("resumed done")])
    llm = LLMClient(
        tiers={"standard": Tier(name="standard", provider=fp, model="f", max_tokens=64)},
        db=db,
        budget=BudgetConfig(),
    )
    convo = ConversationStore(db)
    router = Router(
        llm=llm, registry=registry, convo=convo, db=db, config=RouterConfig(max_tools=2)
    )
    registry.tools["mcp_a"] = _mcp_tool("mcp_a", "banana fetch data alpha")
    registry.tools["mcp_b"] = _mcp_tool("mcp_b", "completely unrelated beta")
    router.refresh_tool_specs()

    r1 = await router.handle(channel="t:1", text="banana fetch", user_id="u1")
    assert r1.pending_confirmation_id is not None
    offered_iter0 = {s.name for s in fp.calls[0]["tools"]}
    assert offered_iter0 == {"wipe_log", "mcp_a"}

    del registry.tools["mcp_a"]  # the server dropped it before resume

    r2 = await router.resolve_confirmation(r1.pending_confirmation_id, approved=True, user_id="u1")
    assert r2.text == "resumed done"
    offered_iter1 = {s.name for s in fp.calls[1]["tools"]}
    assert offered_iter1 == {"wipe_log"}


# -- compaction -----------------------------------------------------------


def make_router_with_compaction(tmp_path, standard_script, compact_script, config=None):
    url = f"sqlite+aiosqlite:///{tmp_path}/rc.db"
    upgrade_core(url)
    db = Database(url)
    fp_standard = FakeProvider(standard_script)
    fp_compact = FakeProvider(compact_script)
    llm = LLMClient(
        tiers={
            "standard": Tier(name="standard", provider=fp_standard, model="f", max_tokens=64),
            "compact": Tier(name="compact", provider=fp_compact, model="f", max_tokens=64),
        },
        db=db,
        budget=BudgetConfig(),
    )
    registry = Registry([make_app()])
    convo = ConversationStore(db)
    compactor = Compactor(llm=llm, db=db, tier="compact")
    router = Router(
        llm=llm,
        registry=registry,
        convo=convo,
        db=db,
        config=config or RouterConfig(window_tokens=10),
        compactor=compactor,
    )
    return router, fp_standard, fp_compact, db, convo


async def test_compaction_prepends_summary_once_per_turn_within_budget(tmp_path) -> None:
    """A long conversation that overflows window_tokens: the model's window
    starts with the system message then the summary as a user message, the
    summarizer runs exactly once even across two _loop iterations (a tool
    call followed by the final reply), and the window handed to the model
    stays within window_tokens including the summary's own cost."""
    config = RouterConfig(window_tokens=60)
    router, fp_standard, fp_compact, db, convo = make_router_with_compaction(
        tmp_path,
        [
            fake_tool_call("log_workout", {"exercise": "bench", "reps": 5}),
            fake_text("final answer"),
        ],
        [fake_text("SUMMARY: recap of the earlier discussion.")],
        config=config,
    )
    conv_id = await convo.get_or_create("t:1")
    filler = "filler " * 20
    for i in range(5):
        await convo.append(conv_id, Message(role="user", text=f"prior user {i} {filler}"))
        await convo.append(conv_id, Message(role="assistant", text=f"prior assistant {i} {filler}"))

    reply = await router.handle(channel="t:1", text="hi", user_id="u1")
    assert reply.text == "final answer"

    # exactly one summarizer call across both _loop iterations
    assert len(fp_compact.calls) == 1
    assert fp_compact.calls[0]["tools"] is None

    first_call_messages = fp_standard.calls[0]["messages"]
    assert first_call_messages[0].role == "system"
    assert first_call_messages[1].role == "user"
    assert "SUMMARY: recap of the earlier discussion." in first_call_messages[1].text
    assert "background context" in first_call_messages[1].text  # data framing, not an instruction

    total = sum(message_tokens(m) for m in first_call_messages if m.role != "system")
    assert total <= config.window_tokens

    async with db.session() as s:
        rows = (
            (await s.execute(select(Summary).where(Summary.conversation_id == conv_id)))
            .scalars()
            .all()
        )
    assert len(rows) == 1  # exactly one row written for the whole turn
    await db.dispose()


async def test_initial_taint_seeded_from_newest_summary_in_default_turn_mode(tmp_path) -> None:
    """`_loop` seeds turn_tainted from the newest summary's flag under the
    default taint_mode="turn" -- not just "window" -- even when nothing
    dropped or summarized this turn."""
    config = RouterConfig(window_tokens=10, taint_mode="turn")
    router, fp_standard, fp_compact, db, convo = make_router_with_compaction(
        tmp_path,
        [fake_tool_call("log_workout", {"exercise": "bench", "reps": 5})],
        [],
        config=config,
    )
    conv_id = await convo.get_or_create("t:1")
    async with db.session() as s:
        s.add(
            Summary(
                conversation_id=conv_id,
                up_to_message_id=0,
                text="earlier mcp-origin output summarized here",
                tainted=True,
            )
        )

    reply = await router.handle(channel="t:1", text="log it", user_id="u1")

    assert reply.pending_confirmation_id is not None
    assert "Confirm: run log_workout" in reply.text
    assert fp_compact.calls == []  # nothing dropped this turn -- no summarizer call
    await db.dispose()


async def test_oversized_newest_turn_still_sent_despite_compaction_accounting(tmp_path) -> None:
    """build_window's newest-turn-is-non-negotiable rule (window.py) means a
    single turn larger than the whole budget is still sent whole, with or
    without compaction competing for the same budget. This is the boundary
    the summary-cost subtraction does NOT smooth over: it only keeps
    compaction from adding pressure beyond an uncompacted turn, and must not
    crash or mis-drop when the remaining budget can't hold the oversized
    turn either."""
    config = RouterConfig(window_tokens=10)
    huge_text = "gigantic user turn " * 200  # unmistakably larger than window_tokens alone
    router, fp_standard, fp_compact, db, convo = make_router_with_compaction(
        tmp_path,
        [fake_text("final answer")],
        [fake_text("SUMMARY: recap.")],
        config=config,
    )
    conv_id = await convo.get_or_create("t:1")
    filler = "filler " * 20
    for i in range(3):
        await convo.append(conv_id, Message(role="user", text=f"prior user {i} {filler}"))
        await convo.append(conv_id, Message(role="assistant", text=f"prior assistant {i} {filler}"))

    reply = await router.handle(channel="t:1", text=huge_text, user_id="u1")
    assert reply.text == "final answer"

    messages = fp_standard.calls[0]["messages"]
    assert messages[0].role == "system"
    assert messages[1].role == "user" and "SUMMARY: recap." in messages[1].text
    # the oversized newest turn is still sent whole, not truncated or dropped
    assert messages[-1].role == "user" and messages[-1].text == huge_text

    async with db.session() as s:
        rows = (
            (await s.execute(select(Summary).where(Summary.conversation_id == conv_id)))
            .scalars()
            .all()
        )
    assert len(rows) == 1  # accounting still wrote exactly one summary row, no crash
    await db.dispose()


async def test_compaction_recompute_gap_still_taints(tmp_path) -> None:
    """When the summary-cost subtraction forces the second `build_window`
    call to drop MORE than the span the summary itself covers, the extra
    ("gap") turns are neither in the rebuilt window nor covered by the
    summary's own taint flag -- the summary was computed against the
    ORIGINAL, smaller dropped count. If an mcp-origin call sits in that
    gap, a native mutation later in the same turn must still be
    confirm-gated."""
    url = f"sqlite+aiosqlite:///{tmp_path}/gap.db"
    upgrade_core(url)
    db = Database(url)
    fp_standard = FakeProvider([fake_tool_call("log_workout", {"exercise": "bench", "reps": 5})])
    fp_compact = FakeProvider([fake_text("s")])
    llm = LLMClient(
        tiers={
            "standard": Tier(name="standard", provider=fp_standard, model="f", max_tokens=64),
            "compact": Tier(name="compact", provider=fp_compact, model="f", max_tokens=64),
        },
        db=db,
        budget=BudgetConfig(),
    )
    registry = Registry([make_app()])
    registry.tools["web__fetch"] = _mcp_tool("web__fetch", "Fetch a page.")
    convo = ConversationStore(db)
    compactor = Compactor(llm=llm, db=db, tier="compact")
    config = RouterConfig(window_tokens=50, taint_mode="window")
    router = Router(
        llm=llm, registry=registry, convo=convo, db=db, config=config, compactor=compactor
    )
    conv_id = await convo.get_or_create("t:1")

    # turn 0 (oldest): large plain filler -- the only thing a FULL-budget
    # build_window drops, so the summary's span (and its untainted flag)
    # covers only this turn.
    await convo.append(conv_id, Message(role="user", text="x" * 200))
    # turn 1 (the gap): an mcp-origin call. Fits under the full budget
    # (so it survives into the first, untainted-summary build) but not
    # under the smaller budget left after the summary's own cost is
    # subtracted.
    await convo.append(conv_id, Message(role="user", text="y" * 40))
    await convo.append_many(
        conv_id,
        [
            Message(role="assistant", tool_calls=[ToolCall(id="g1", name="web__fetch", args={})]),
            Message(role="tool", text="z" * 40, tool_call_id="g1"),
        ],
    )
    # turn 2: small filler that keeps the newest turn company under the
    # reduced, post-summary budget.
    await convo.append(conv_id, Message(role="user", text="w" * 40))
    await convo.append(conv_id, Message(role="assistant", text="ok"))

    reply = await router.handle(channel="t:1", text="log it", user_id="u1")

    assert len(fp_compact.calls) == 1  # exactly one summarizer call
    async with db.session() as s:
        rows = (
            (await s.execute(select(Summary).where(Summary.conversation_id == conv_id)))
            .scalars()
            .all()
        )
    assert len(rows) == 1
    assert rows[0].tainted is False  # the summary's own span (turn 0) has no mcp call

    # The gap (turn 1's mcp call) is invisible in the rebuilt window and
    # uncovered by the summary's taint flag -- the mutation must still be
    # confirm-gated.
    assert reply.pending_confirmation_id is not None
    assert CALLS == []  # log_workout did not run
    await db.dispose()
