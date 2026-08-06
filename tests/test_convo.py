import asyncio
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from dudamel.convo import ConversationStore
from dudamel.db import Database
from dudamel.llm.types import Message, ToolCall
from dudamel.migrate import upgrade_core
from dudamel.models_core import Conversation
from dudamel.models_core import Message as MessageRow


@pytest.fixture
async def store(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/v.db"
    upgrade_core(url)
    d = Database(url)
    yield ConversationStore(d)
    await d.dispose()


async def test_get_or_create_is_stable(store: ConversationStore) -> None:
    a = await store.get_or_create("telegram:1")
    b = await store.get_or_create("telegram:1")
    c = await store.get_or_create("web:sess9")
    assert a == b and a != c


async def test_append_and_recent_roundtrip(store: ConversationStore) -> None:
    cid = await store.get_or_create("t:1")
    assert await store.append(cid, Message(role="user", text="hi"))
    assert await store.append(
        cid,
        Message(role="assistant", tool_calls=[ToolCall(id="a", name="t", args={"x": 1})]),
    )
    assert await store.append(cid, Message(role="tool", text="ok", tool_call_id="a"))
    msgs = await store.recent(cid)
    assert [m.role for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[1].tool_calls[0].args == {"x": 1}
    assert msgs[2].tool_call_id == "a"


async def test_client_msg_id_dedupe(store: ConversationStore) -> None:
    cid = await store.get_or_create("t:1")
    assert await store.append(cid, Message(role="user", text="x"), client_msg_id="u42")
    assert not await store.append(cid, Message(role="user", text="x"), client_msg_id="u42")
    assert len(await store.recent(cid)) == 1


async def test_concurrent_get_or_create_same_new_channel(store: ConversationStore) -> None:
    """Regression test for a TOCTOU race: two get_or_create() calls for the
    same brand-new channel both SELECT and find nothing before either
    commits an INSERT. Each racer's session boundary is gated by a shared
    asyncio.Event so neither can reach its INSERT until *both* have
    completed their SELECT -- exactly the interleaving the raw
    IntegrityError bug required. Proves: (1) the loser recovers the
    winner's id instead of raising, (2) exactly one Conversation row for
    the channel ends up persisted."""
    channel = "race:new-channel"
    real_session = store._db.session
    both_selected = asyncio.Event()
    select_count = 0

    def make_racer() -> ConversationStore:
        calls = 0

        @asynccontextmanager
        async def gated_session():
            nonlocal calls, select_count
            calls += 1
            is_select_call = calls == 1
            if not is_select_call:
                await both_selected.wait()  # hold the INSERT until both raced
            async with real_session() as s:
                yield s
            if is_select_call:
                select_count += 1
                if select_count == 2:
                    both_selected.set()

        fake_db = type("_GatedDB", (), {"session": staticmethod(gated_session)})()
        return ConversationStore(fake_db)

    a, b = make_racer(), make_racer()
    ids = await asyncio.gather(a.get_or_create(channel), b.get_or_create(channel))
    assert ids[0] == ids[1]

    async with store._db.session() as s:
        rows = (
            (await s.execute(select(Conversation).where(Conversation.channel == channel)))
            .scalars()
            .all()
        )
    assert len(rows) == 1
    assert rows[0].id == ids[0]


async def test_message_client_msg_id_unique_constraint(store: ConversationStore) -> None:
    """Bypasses append()'s in-app pre-check by inserting two MessageRow rows
    with the same (conversation_id, client_msg_id) via raw sessions. The
    second raw insert must raise IntegrityError, proving migration 0003's
    unique index is a real DB-level constraint and not just the app-level
    check."""
    cid = await store.get_or_create("t:race-append")
    dup_msg = Message(role="user", text="dup").to_dict()

    async with store._db.session() as s:
        s.add(MessageRow(conversation_id=cid, role="user", content=dup_msg, client_msg_id="dup1"))

    with pytest.raises(IntegrityError):
        async with store._db.session() as s:
            s.add(
                MessageRow(conversation_id=cid, role="user", content=dup_msg, client_msg_id="dup1")
            )


async def test_append_concurrent_race_backstopped(store: ConversationStore) -> None:
    """Reproduces a genuine append() race: two concurrent store.append()
    calls carrying the same client_msg_id are gated -- via a shared
    asyncio.Event, mirroring test_concurrent_get_or_create_same_new_channel
    -- so that BOTH complete the dedupe pre-check SELECT (each seeing no
    existing row) and stage their INSERT before EITHER racer's session is
    allowed to commit. Structural argument that makes this airtight: since
    both passed the pre-check, the False in the result can ONLY have come
    from append()'s `except IntegrityError: return False` branch -- the
    pre-check was, by construction, blind to the race."""
    cid = await store.get_or_create("t:race-append-concurrent")
    real_session = store._db.session
    both_ready = asyncio.Event()
    done_count = 0

    def make_racer() -> ConversationStore:
        @asynccontextmanager
        async def gated_session():
            nonlocal done_count
            async with real_session() as s:
                yield s
                # append()'s body (dedupe SELECT + staged INSERT) has just
                # run; hold the commit until both racers get this far so
                # neither's SELECT can see the other's not-yet-committed row.
                done_count += 1
                if done_count < 2:
                    await both_ready.wait()
                else:
                    both_ready.set()

        fake_db = type("_GatedDB", (), {"session": staticmethod(gated_session)})()
        return ConversationStore(fake_db)

    a, b = make_racer(), make_racer()
    results = await asyncio.gather(
        a.append(cid, Message(role="user", text="race"), client_msg_id="race1"),
        b.append(cid, Message(role="user", text="race"), client_msg_id="race1"),
    )
    assert sorted(results) == [False, True]

    async with store._db.session() as s:
        rows = (
            (
                await s.execute(
                    select(MessageRow).where(
                        MessageRow.conversation_id == cid,
                        MessageRow.client_msg_id == "race1",
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1


async def test_dedupe_is_per_conversation(store: ConversationStore) -> None:
    c1 = await store.get_or_create("t:1")
    c2 = await store.get_or_create("t:2")
    assert await store.append(c1, Message(role="user", text="x"), client_msg_id="m1")
    assert await store.append(c2, Message(role="user", text="x"), client_msg_id="m1")


async def test_recent_respects_limit(store: ConversationStore) -> None:
    cid = await store.get_or_create("t:3")
    for i in range(10):
        await store.append(cid, Message(role="user", text=str(i)))
    msgs = await store.recent(cid, limit=4)
    assert [m.text for m in msgs] == ["6", "7", "8", "9"]  # last 4, chronological


async def test_append_many_preserves_order(store: ConversationStore) -> None:
    cid = await store.get_or_create("t:batch-order")
    await store.append(cid, Message(role="user", text="go"))
    await store.append_many(
        cid,
        [
            Message(role="assistant", tool_calls=[ToolCall(id="a", name="t", args={})]),
            Message(role="tool", text="first", tool_call_id="a"),
            Message(role="tool", text="second", tool_call_id="a"),
        ],
    )
    msgs = await store.recent(cid)
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "tool"]
    assert [m.text for m in msgs[2:]] == ["first", "second"]


async def test_append_many_is_all_or_nothing(store: ConversationStore) -> None:
    """A failure partway through the batch must leave ZERO rows from it --
    that is the whole point of the primitive. Injected by making the second
    message's to_dict() raise, after the first has already been added to the
    session."""
    cid = await store.get_or_create("t:batch-atomic")
    before = len(await store.recent(cid))

    class _Exploding(Message):
        def to_dict(self):  # type: ignore[override]
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await store.append_many(
            cid,
            [
                Message(role="assistant", text="kept?"),
                _Exploding(role="tool", text="never", tool_call_id="x"),
            ],
        )

    assert len(await store.recent(cid)) == before


async def test_append_many_empty_is_a_noop(store: ConversationStore) -> None:
    cid = await store.get_or_create("t:batch-empty")
    await store.append_many(cid, [])
    assert await store.recent(cid) == []
