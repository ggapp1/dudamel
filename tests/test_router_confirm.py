from datetime import timedelta

import pytest
from sqlalchemy import select

from dudamel import App
from dudamel.config import BudgetConfig, RouterConfig
from dudamel.convo import ConversationStore
from dudamel.db import Database
from dudamel.exceptions import LLMError, UnknownToolOutcome
from dudamel.llm.client import LLMClient, Tier
from dudamel.llm.testing import FakeProvider, fake_text, fake_tool_call
from dudamel.llm.types import Completion, Message, ToolCall, Usage
from dudamel.migrate import upgrade_core
from dudamel.models_core import PendingConfirmation
from dudamel.registry import Registry
from dudamel.router import ChatReply, Router, _confirmed_error_text, _utcnow

DELETED: list[str] = []
MUTATED: list[str] = []
READS: list[str] = []


def make_app() -> App:
    app = App("gym", description="d")

    @app.tool(confirm=True)
    async def wipe_log(reason: str) -> str:
        """Delete the whole workout log."""
        DELETED.append(reason)
        return "wiped"

    @app.tool(read_only=True)
    async def cloud_fetch() -> str:
        """Fetch remote data (marked MCP-origin in the taint test)."""
        READS.append("fetch")
        return "cloud data"

    @app.tool
    async def set_pref(value: str) -> str:
        """A native mutation: not read-only, not confirm — taint-gated only."""
        MUTATED.append(value)
        return f"set {value}"

    return app


def build(tmp_path, script):
    url = f"sqlite+aiosqlite:///{tmp_path}/cf.db"
    upgrade_core(url)
    db = Database(url)
    fp = FakeProvider(script)
    llm = LLMClient(
        tiers={"standard": Tier(name="standard", provider=fp, model="f", max_tokens=64)},
        db=db,
        budget=BudgetConfig(),
    )
    registry = Registry([make_app()])
    convo = ConversationStore(db)
    router = Router(llm=llm, registry=registry, convo=convo, db=db, config=RouterConfig())
    return router, fp, db, convo, registry


@pytest.fixture(autouse=True)
def _clear():
    DELETED.clear()
    MUTATED.clear()
    READS.clear()


async def test_suspend_then_approve_executes_and_resumes(tmp_path) -> None:
    script = [fake_tool_call("wipe_log", {"reason": "spring cleaning"}), fake_text("All wiped!")]
    router, fp, db, convo, registry = build(tmp_path, script)
    r1 = await router.handle(channel="t:1", text="wipe it", user_id="u1")
    assert r1.pending_confirmation_id and "Confirm" in r1.text
    assert DELETED == []  # nothing ran yet
    cid = await convo.get_or_create("t:1")
    assert [m.role for m in await convo.recent(cid)] == ["user"]  # no dangling turn
    r2 = await router.resolve_confirmation(r1.pending_confirmation_id, approved=True, user_id="u1")
    assert r2.text == "All wiped!" and DELETED == ["spring cleaning"]
    roles = [m.role for m in await convo.recent(cid)]
    assert roles == ["user", "assistant", "tool", "assistant"]  # intact history
    await db.dispose()


async def test_deny_lets_model_see_decline(tmp_path) -> None:
    script = [fake_tool_call("wipe_log", {"reason": "x"}), fake_text("Okay, cancelled.")]
    router, fp, db, convo, _ = build(tmp_path, script)
    r1 = await router.handle(channel="t:1", text="wipe", user_id="u1")
    r2 = await router.resolve_confirmation(r1.pending_confirmation_id, approved=False, user_id="u1")
    assert r2.text == "Okay, cancelled." and DELETED == []
    declined = [m for m in fp.calls[1]["messages"] if m.role == "tool"][0]
    assert declined.is_error and "declined" in declined.text
    await db.dispose()


async def test_wrong_user_cannot_resolve(tmp_path) -> None:
    script = [fake_tool_call("wipe_log", {"reason": "x"})]
    router, fp, db, convo, _ = build(tmp_path, script)
    r1 = await router.handle(channel="t:1", text="wipe", user_id="u1")
    r2 = await router.resolve_confirmation(
        r1.pending_confirmation_id, approved=True, user_id="intruder"
    )
    assert "requester" in r2.text and DELETED == []
    async with db.session() as s:
        row = (await s.execute(select(PendingConfirmation))).scalar_one()
    assert row.status == "pending"  # untouched
    await db.dispose()


async def test_expired_confirmation_declines_without_model_call(tmp_path) -> None:
    script = [fake_tool_call("wipe_log", {"reason": "x"})]
    router, fp, db, convo, _ = build(tmp_path, script)
    r1 = await router.handle(channel="t:1", text="wipe", user_id="u1")
    async with db.session() as s:
        row = (await s.execute(select(PendingConfirmation))).scalar_one()
        row.expires_at = _utcnow() - timedelta(seconds=1)
    r2 = await router.resolve_confirmation(r1.pending_confirmation_id, approved=True, user_id="u1")
    assert "expired" in r2.text.lower() and DELETED == []
    assert len(fp.calls) == 1  # no further model call
    cid = await convo.get_or_create("t:1")
    roles = [m.role for m in await convo.recent(cid)]
    assert roles == ["user", "assistant", "tool"]  # declined result appended
    await db.dispose()


async def test_new_message_auto_declines_pending(tmp_path) -> None:
    script = [fake_tool_call("wipe_log", {"reason": "x"}), fake_text("hello!")]
    router, fp, db, convo, _ = build(tmp_path, script)
    await router.handle(channel="t:1", text="wipe", user_id="u1")
    r2 = await router.handle(channel="t:1", text="actually, hi", user_id="u1")
    assert r2.text == "hello!"
    async with db.session() as s:
        row = (await s.execute(select(PendingConfirmation))).scalar_one()
    assert row.status == "declined" and DELETED == []
    # exactly 2 model calls: the suspend turn and the new message — none for decline
    assert len(fp.calls) == 2
    cid = await convo.get_or_create("t:1")
    roles = [m.role for m in await convo.recent(cid)]
    assert roles == ["user", "assistant", "tool", "user", "assistant"]
    await db.dispose()


async def test_resolution_survives_router_restart(tmp_path) -> None:
    script = [fake_tool_call("wipe_log", {"reason": "persist"})]
    router, fp, db, convo, registry = build(tmp_path, script)
    r1 = await router.handle(channel="t:1", text="wipe", user_id="u1")
    # fresh Router over the same DB (new FakeProvider for the resume completion)
    fp2 = FakeProvider([fake_text("done after restart")])
    llm2 = LLMClient(
        tiers={"standard": Tier(name="standard", provider=fp2, model="f", max_tokens=64)},
        db=db,
        budget=BudgetConfig(),
    )
    router2 = Router(llm=llm2, registry=registry, convo=convo, db=db, config=RouterConfig())
    r2 = await router2.resolve_confirmation(r1.pending_confirmation_id, approved=True, user_id="u1")
    assert r2.text == "done after restart" and DELETED == ["persist"]
    await db.dispose()


async def test_duplicate_of_confirming_message_does_not_decline_pending(tmp_path) -> None:
    """I2: an interface retry (same client_msg_id) of the very message that
    created a pending confirmation must not auto-decline that confirmation
    on its way to being deduped."""
    script = [fake_tool_call("wipe_log", {"reason": "x"}), fake_text("All wiped!")]
    router, fp, db, convo, _ = build(tmp_path, script)
    r1 = await router.handle(channel="t:1", text="wipe it", user_id="u1", client_msg_id="m1")
    assert r1.pending_confirmation_id and "Confirm" in r1.text

    r2 = await router.handle(channel="t:1", text="wipe it", user_id="u1", client_msg_id="m1")
    assert r2 == ChatReply(text="")

    async with db.session() as s:
        row = (await s.execute(select(PendingConfirmation))).scalar_one()
    assert row.status == "pending"  # NOT auto-declined by the retry
    assert len(fp.calls) == 1  # no second model call from the retry

    r3 = await router.resolve_confirmation(r1.pending_confirmation_id, approved=True, user_id="u1")
    assert r3.text == "All wiped!" and DELETED == ["x"]
    await db.dispose()


async def test_unknown_confirmation_id(tmp_path) -> None:
    router, fp, db, convo, _ = build(tmp_path, [])
    r = await router.resolve_confirmation("nope", approved=True, user_id="u1")
    assert "unknown" in r.text.lower()
    await db.dispose()


# --- resume honesty: what a post-suspension model failure reports -----------


async def test_post_approve_llm_error_reports_action_completed(tmp_path) -> None:
    """loop_state carries executed_any across the suspension gap: after
    approve, the confirmed tool ran, so a model failure on resume must
    honestly report the action completed — not that the model was
    unavailable."""
    script = [
        fake_tool_call("wipe_log", {"reason": "x"}),
        LLMError("connection reset", retryable=True),
    ]
    router, fp, db, convo, _ = build(tmp_path, script)
    r1 = await router.handle(channel="t:1", text="wipe", user_id="u1")
    r2 = await router.resolve_confirmation(r1.pending_confirmation_id, approved=True, user_id="u1")
    assert "completed the action" in r2.text and DELETED == ["x"]
    assert len(fp.calls) == 2  # suspend turn + failed resume completion
    await db.dispose()


async def test_approve_with_failing_tool_reports_failure_honestly(tmp_path) -> None:
    """The confirmed tool RAISES and there is no prior success, so a
    post-resume LLMError must report the model as unavailable — NOT falsely
    claim "I completed the action(s)"."""

    app = App("gym", description="d")

    @app.tool(confirm=True)
    async def wipe_log(reason: str) -> str:
        """Delete the whole workout log."""
        raise RuntimeError("disk full")

    script = [
        fake_tool_call("wipe_log", {"reason": "x"}),
        LLMError("provider down", retryable=True),
    ]
    url = f"sqlite+aiosqlite:///{tmp_path}/cf.db"
    upgrade_core(url)
    db = Database(url)
    fp = FakeProvider(script)
    llm = LLMClient(
        tiers={"standard": Tier(name="standard", provider=fp, model="f", max_tokens=64)},
        db=db,
        budget=BudgetConfig(),
    )
    registry = Registry([app])
    convo = ConversationStore(db)
    router = Router(llm=llm, registry=registry, convo=convo, db=db, config=RouterConfig())

    r1 = await router.handle(channel="t:1", text="wipe it", user_id="u1")
    assert r1.pending_confirmation_id and "Confirm" in r1.text
    r2 = await router.resolve_confirmation(r1.pending_confirmation_id, approved=True, user_id="u1")

    assert "unavailable" in r2.text
    assert "completed the action" not in r2.text
    assert DELETED == []

    cid = await convo.get_or_create("t:1")
    history = await convo.recent(cid)
    assert [m.role for m in history] == ["user", "assistant", "tool"]
    tool_msg = history[-1]
    assert tool_msg.is_error
    assert "disk full" in tool_msg.text
    await db.dispose()


async def test_taint_survives_suspension_gap(tmp_path) -> None:
    """loop_state carries turn_tainted across the suspension gap: an MCP
    result in the suspended batch taints the turn; after approving the first
    mutation, a LATER native mutation (in an mcp-free batch) must still be
    confirm-gated — proven by a SECOND pending confirmation, not execution."""
    turn1 = Completion(
        message=Message(
            role="assistant",
            tool_calls=[
                ToolCall(id="mcp1", name="cloud_fetch", args={}),
                ToolCall(id="mut1", name="set_pref", args={"value": "a"}),
            ],
        ),
        usage=Usage(1, 1),
        stop_reason="tool_calls",
    )
    turn2 = fake_tool_call("set_pref", {"value": "b"}, id="mut2")
    router, fp, db, convo, registry = build(tmp_path, [turn1, turn2])
    registry.tools["cloud_fetch"].origin = "mcp"  # simulate an MCP-provided read

    r1 = await router.handle(channel="t:1", text="do it", user_id="u1")
    assert r1.pending_confirmation_id  # set_pref gated by batch_has_mcp
    assert MUTATED == [] and READS == ["fetch"]  # mcp read ran, mutation held

    r2 = await router.resolve_confirmation(r1.pending_confirmation_id, approved=True, user_id="u1")
    # approving "a" runs it, then the resumed loop hits set_pref("b"): still
    # gated because the carried-over taint outlives the suspension — suspends again.
    assert MUTATED == ["a"]  # "b" NOT executed
    assert r2.pending_confirmation_id and r2.pending_confirmation_id != r1.pending_confirmation_id
    async with db.session() as s:
        rows = (await s.execute(select(PendingConfirmation))).scalars().all()
    assert sorted(row.status for row in rows) == ["confirmed", "pending"]
    second = next(row for row in rows if row.status == "pending")
    assert second.tool == "set_pref" and second.args == {"value": "b"}
    await db.dispose()


def test_confirmed_tool_failure_message_does_not_call_an_unknown_outcome_failed() -> None:
    """On the confirmed path the router prefixes its own wording, so an
    indeterminate outcome would otherwise be announced to the model with the
    exact word the tool text was written to avoid."""
    indeterminate = _confirmed_error_text(
        "files__write",
        UnknownToolOutcome(
            "mcp tool files__write: the call timed out after 30s and the outcome is "
            "UNKNOWN -- the server may still be completing it. Do not retry "
            "automatically; check the server's state first."
        ),
    )
    assert "failed" not in indeterminate
    assert "UNKNOWN" in indeterminate and "Do not retry" in indeterminate
    # Anything else keeps the plain, unambiguous failure wording, including
    # the exception type -- an ordinary error must still read as an error.
    ordinary = _confirmed_error_text("wipe_log", RuntimeError("disk is on fire"))
    assert ordinary == "confirmed tool wipe_log failed: RuntimeError: disk is on fire"


async def test_confirmed_indeterminate_outcome_reaches_the_model_unfailed(tmp_path) -> None:
    """The end-to-end version of the test above: what the model is actually
    handed after approving a tool whose outcome turned out to be unknown."""
    script = [fake_tool_call("wipe_log", {"reason": "x"}), fake_text("I'll check first.")]
    router, fp, db, convo, registry = build(tmp_path, script)

    async def indeterminate(reason: str) -> str:
        raise UnknownToolOutcome(
            "mcp tool wipe_log: the server connection died during the call and the "
            "outcome is UNKNOWN -- it may or may not have taken effect. Do not retry "
            "automatically; check the server's state first."
        )

    registry.tools["wipe_log"].fn = indeterminate
    r1 = await router.handle(channel="t:1", text="wipe", user_id="u1")
    await router.resolve_confirmation(r1.pending_confirmation_id, approved=True, user_id="u1")
    result = [m for m in fp.calls[1]["messages"] if m.role == "tool"][0]
    assert result.is_error  # still an error result; just not a *failed* one
    assert "failed" not in result.text
    assert "UNKNOWN" in result.text and "Do not retry" in result.text
    await db.dispose()
