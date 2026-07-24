import pytest

from dudamel.convo import ConversationStore
from dudamel.db import Database
from dudamel.llm.types import Message, ToolCall
from dudamel.migrate import upgrade_core


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
