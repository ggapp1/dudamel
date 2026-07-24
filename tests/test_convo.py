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
    """Reproduces the reviewer's TOCTOU repro: two get_or_create() calls for
    the same brand-new channel both SELECT and find nothing before either
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


async def test_append_duplicate_race_backstopped(store: ConversationStore) -> None:
    """Bypasses append()'s in-app pre-check by inserting two MessageRow rows
    with the same (conversation_id, client_msg_id) via raw sessions. The
    second raw insert must raise IntegrityError, proving migration 0003's
    unique index is a real DB-level constraint and not just the app-level
    check. Then confirms append() itself surfaces that as `False` rather
    than letting the exception escape."""
    cid = await store.get_or_create("t:race-append")
    dup_msg = Message(role="user", text="dup").to_dict()

    async with store._db.session() as s:
        s.add(MessageRow(conversation_id=cid, role="user", content=dup_msg, client_msg_id="dup1"))

    with pytest.raises(IntegrityError):
        async with store._db.session() as s:
            s.add(
                MessageRow(conversation_id=cid, role="user", content=dup_msg, client_msg_id="dup1")
            )

    assert not await store.append(cid, Message(role="user", text="dup"), client_msg_id="dup1")


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
