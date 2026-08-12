import logging
from pathlib import Path

import pytest
from sqlalchemy import select

from dudamel.compaction import _KEEP_PER_CONVERSATION, Compactor
from dudamel.config import BudgetConfig
from dudamel.db import Database
from dudamel.exceptions import BudgetExceededError, LLMError
from dudamel.llm.client import LLMClient, Tier
from dudamel.llm.testing import FakeProvider, fake_text
from dudamel.llm.types import Message
from dudamel.migrate import upgrade_core
from dudamel.models_core import Conversation, LlmCall, Summary
from dudamel.models_core import Message as MessageRow


async def _make(
    tmp_path: Path, script, *, max_tokens: int = 111, budget: BudgetConfig | None = None
):
    url = f"sqlite+aiosqlite:///{tmp_path}/c.db"
    upgrade_core(url)
    db = Database(url)
    fp = FakeProvider(script)
    llm = LLMClient(
        tiers={"compact": Tier(name="compact", provider=fp, model="m", max_tokens=max_tokens)},
        db=db,
        budget=budget if budget is not None else BudgetConfig(),
    )
    compactor = Compactor(llm=llm, db=db, tier="compact")
    async with db.session() as s:
        conv = Conversation(channel="t:1")
        s.add(conv)
        await s.flush()
        conv_id = conv.id
    return compactor, fp, db, conv_id


async def _seed_messages(db: Database, conv_id: int, n: int) -> list[Message]:
    """Insert n plain user messages and return the equivalent Message list,
    matching what ConversationStore.recent() would return."""
    async with db.session() as s:
        for i in range(n):
            content = {"role": "user", "text": f"m{i}"}
            s.add(MessageRow(conversation_id=conv_id, role="user", content=content))
    return [Message(role="user", text=f"m{i}") for i in range(n)]


async def _message_ids(db: Database, conv_id: int) -> list[int]:
    async with db.session() as s:
        stmt = select(MessageRow.id).where(MessageRow.conversation_id == conv_id)
        rows = (await s.execute(stmt.order_by(MessageRow.id))).scalars().all()
    return list(rows)


async def _summary_rows(db: Database, conv_id: int) -> list[Summary]:
    async with db.session() as s:
        stmt = select(Summary).where(Summary.conversation_id == conv_id).order_by(Summary.id)
        return list((await s.execute(stmt)).scalars().all())


async def test_summarizes_dropped_span_and_writes_a_row(tmp_path: Path) -> None:
    compactor, fp, db, conv_id = await _make(tmp_path, [fake_text("the gist")])
    history = await _seed_messages(db, conv_id, 5)
    record = await compactor.maybe_compact(
        conv_id, history, dropped=3, turn_key="turn-1", dropped_tainted=False
    )
    assert record is not None
    assert record.text == "the gist"
    assert record.tainted is False
    rows = await _summary_rows(db, conv_id)
    assert len(rows) == 1
    await db.dispose()


async def test_summarizer_call_carries_tools_none_and_pinned_max_tokens(tmp_path: Path) -> None:
    compactor, fp, db, conv_id = await _make(tmp_path, [fake_text("gist")], max_tokens=222)
    history = await _seed_messages(db, conv_id, 4)
    await compactor.maybe_compact(conv_id, history, dropped=2, turn_key="t", dropped_tainted=False)
    assert len(fp.calls) == 1
    assert fp.calls[0]["tools"] is None
    assert fp.calls[0]["max_tokens"] == 222
    await db.dispose()


async def test_once_per_turn_idempotence(tmp_path: Path) -> None:
    """Two calls sharing a turn_key must summarize at most once -- the
    iteration-cap feedback loop the binding requirement guards against."""
    compactor, fp, db, conv_id = await _make(tmp_path, [fake_text("only once")])
    history = await _seed_messages(db, conv_id, 5)
    first = await compactor.maybe_compact(
        conv_id, history, dropped=3, turn_key="same-turn", dropped_tainted=False
    )
    second = await compactor.maybe_compact(
        conv_id, history, dropped=3, turn_key="same-turn", dropped_tainted=False
    )
    assert len(fp.calls) == 1
    assert first is not None and second is not None
    assert first.id == second.id


async def test_different_turn_keys_each_get_their_own_check(tmp_path: Path) -> None:
    """A different turn_key is free to summarize again -- once-per-turn, not
    once-per-conversation-forever -- but a span already covered by the
    newest summary is reused rather than resummarized."""
    compactor, fp, db, conv_id = await _make(tmp_path, [fake_text("gist")])
    history = await _seed_messages(db, conv_id, 5)
    first = await compactor.maybe_compact(
        conv_id, history, dropped=3, turn_key="turn-a", dropped_tainted=False
    )
    second = await compactor.maybe_compact(
        conv_id, history, dropped=3, turn_key="turn-b", dropped_tainted=False
    )
    assert len(fp.calls) == 1  # second call's span is already covered
    assert first.id == second.id
    await db.dispose()


async def test_reuses_newest_summary_when_it_already_covers_the_span(tmp_path: Path) -> None:
    compactor, fp, db, conv_id = await _make(tmp_path, [])
    history = await _seed_messages(db, conv_id, 5)
    ids = await _message_ids(db, conv_id)
    async with db.session() as s:
        s.add(
            Summary(
                conversation_id=conv_id,
                up_to_message_id=ids[-1],
                text="already covered",
                tainted=True,
            )
        )
    record = await compactor.maybe_compact(
        conv_id, history, dropped=2, turn_key="fresh-turn", dropped_tainted=False
    )
    assert record is not None
    assert record.text == "already covered"
    assert record.tainted is True
    assert fp.calls == []  # no summarizer call at all
    await db.dispose()


async def test_taint_comes_from_provenance_not_summarizer_output(tmp_path: Path) -> None:
    """The summarizer text carries no taint markers -- tainted must come
    entirely from the dropped_tainted argument the caller computed."""
    compactor, fp, db, conv_id = await _make(tmp_path, [fake_text("totally clean-looking summary")])
    history = await _seed_messages(db, conv_id, 5)
    record = await compactor.maybe_compact(
        conv_id, history, dropped=3, turn_key="t", dropped_tainted=True
    )
    assert record is not None and record.tainted is True
    await db.dispose()


async def test_fence_stripped_from_summarizer_output(tmp_path: Path) -> None:
    compactor, fp, db, conv_id = await _make(tmp_path, [fake_text("```text\nthe real gist\n```")])
    history = await _seed_messages(db, conv_id, 5)
    record = await compactor.maybe_compact(
        conv_id, history, dropped=3, turn_key="t", dropped_tainted=False
    )
    assert record is not None
    assert record.text == "the real gist"
    await db.dispose()


async def test_length_capped(tmp_path: Path) -> None:
    long_text = "x" * 10_000
    compactor, fp, db, conv_id = await _make(tmp_path, [fake_text(long_text)])
    compactor._max_summary_chars = 100
    history = await _seed_messages(db, conv_id, 5)
    record = await compactor.maybe_compact(
        conv_id, history, dropped=3, turn_key="t", dropped_tainted=False
    )
    assert record is not None
    assert len(record.text) <= 100 + len("…[truncated]")
    await db.dispose()


async def test_failed_summarization_proceeds_uncompacted(tmp_path: Path) -> None:
    """A summarizer failure (including a budget error) must never raise out
    of maybe_compact -- it returns None so the caller proceeds uncompacted."""
    compactor, fp, db, conv_id = await _make(tmp_path, [LLMError("nope")])
    history = await _seed_messages(db, conv_id, 5)
    record = await compactor.maybe_compact(
        conv_id, history, dropped=3, turn_key="t", dropped_tainted=False
    )
    assert record is None
    assert await _summary_rows(db, conv_id) == []
    await db.dispose()


async def test_exhausted_budget_fails_open_and_leaves_the_watermark_where_it_was(
    tmp_path: Path,
) -> None:
    """An exhausted daily token budget is the one summarizer failure a long
    conversation will actually hit, and it is raised by the client before
    the provider is ever reached -- not by the model call itself. It must
    still fail open exactly like any other summarizer failure: nothing
    raised out of `maybe_compact`, nothing to prepend (so the caller
    proceeds with the plain, uncompacted window), and no row written -- so
    the covered-span watermark stays where it was and a later turn, once
    budget is available again, still summarizes that same span rather than
    treating it as already covered.
    """
    budget = BudgetConfig(daily_tokens=100)
    compactor, fp, db, conv_id = await _make(tmp_path, [fake_text("the gist")], budget=budget)
    history = await _seed_messages(db, conv_id, 5)
    # Spend the whole day's budget on an unrelated earlier call.
    async with db.session() as s:
        s.add(LlmCall(tier="standard", provider="fake", model="m", tokens_in=90, tokens_out=10))

    # Pin what the summarizer tier now raises: a `BudgetExceededError` --
    # a `DudamelError` subclass, which is what the fail-open path below is
    # relying on to catch it.
    exhausted = LLMClient(
        tiers={"compact": Tier(name="compact", provider=fp, model="m", max_tokens=111)},
        db=db,
        budget=budget,
    )
    with pytest.raises(BudgetExceededError):
        await exhausted.complete([Message(role="user", text="x")], tier="compact")

    record = await compactor.maybe_compact(
        conv_id, history, dropped=3, turn_key="t1", dropped_tainted=False
    )
    assert record is None  # nothing to prepend: the window falls back uncompacted
    assert fp.calls == []  # the budget guard tripped before any model call
    assert await _summary_rows(db, conv_id) == []
    assert await compactor.newest(conv_id) is None  # watermark not advanced

    # Budget available again on a later turn: the span the failed attempt
    # would have covered is still uncovered, so it gets summarized now.
    budget.daily_tokens = None
    record = await compactor.maybe_compact(
        conv_id, history, dropped=3, turn_key="t2", dropped_tainted=False
    )
    assert record is not None
    assert record.text == "the gist"
    assert record.up_to_message_id == (await _message_ids(db, conv_id))[2]
    await db.dispose()


async def test_empty_cleaned_summary_is_treated_as_failure(tmp_path: Path) -> None:
    """An output that cleans down to nothing (e.g. an empty code fence, or
    pure whitespace) must be treated the same as a summarizer failure: no
    row written, so a later turn tries again instead of reusing an empty
    summary forever."""
    compactor, fp, db, conv_id = await _make(tmp_path, [fake_text("```text\n\n```")])
    history = await _seed_messages(db, conv_id, 5)
    record = await compactor.maybe_compact(
        conv_id, history, dropped=3, turn_key="t", dropped_tainted=False
    )
    assert record is None
    assert await _summary_rows(db, conv_id) == []
    await db.dispose()


async def test_no_dropped_span_returns_none_without_calling_the_model(tmp_path: Path) -> None:
    compactor, fp, db, conv_id = await _make(tmp_path, [])
    history = await _seed_messages(db, conv_id, 5)
    record = await compactor.maybe_compact(
        conv_id, history, dropped=0, turn_key="t", dropped_tainted=False
    )
    assert record is None
    assert fp.calls == []
    await db.dispose()


async def test_newest_n_pruning_keeps_only_the_latest_rows(tmp_path: Path) -> None:
    """Writing more than _KEEP_PER_CONVERSATION summaries deletes the
    oldest, keeping only the newest N."""
    script = [fake_text(f"gist-{i}") for i in range(_KEEP_PER_CONVERSATION + 2)]
    compactor, fp, db, conv_id = await _make(tmp_path, script)
    # Each write needs a strictly larger watermark than the last (the
    # unique index enforces one row per watermark), so grow history and
    # dropped count each round with a fresh turn_key.
    n = 3
    for i in range(_KEEP_PER_CONVERSATION + 2):
        n += 1
        await _seed_messages(db, conv_id, n)
        # re-seed appends messages each round; recompute full history from DB
        async with db.session() as s:
            stmt = select(MessageRow).where(MessageRow.conversation_id == conv_id)
            rows = (await s.execute(stmt.order_by(MessageRow.id))).scalars().all()
        history = [Message.from_dict(r.content) for r in rows]
        await compactor.maybe_compact(
            conv_id,
            history,
            dropped=len(history) - 1,
            turn_key=f"turn-{i}",
            dropped_tainted=False,
        )
    remaining = await _summary_rows(db, conv_id)
    assert len(remaining) == _KEEP_PER_CONVERSATION
    expected = [f"gist-{i}" for i in range(2, _KEEP_PER_CONVERSATION + 2)]
    assert [r.text for r in remaining] == expected
    await db.dispose()


async def test_a_row_landing_between_the_two_reads_proceeds_uncompacted(
    tmp_path: Path, caplog
) -> None:
    """`_watermark_id` lines its ids up positionally with the caller's
    history, which only holds while nothing writes between the two reads --
    the router's per-conversation lock, which `Compactor` neither receives
    nor checks. Simulate the write landing anyway: the ids shift, position
    `dropped - 1` now names a NEWER message, and marking that covered would
    lose it forever. Compaction is best-effort, so the turn proceeds
    uncompacted instead."""
    compactor, fp, db, conv_id = await _make(tmp_path, [fake_text("the gist")])
    history = await _seed_messages(db, conv_id, 5)
    # A second writer (another process, a scheduler job) appends after the
    # caller's `recent()` read produced `history`.
    async with db.session() as s:
        s.add(
            MessageRow(
                conversation_id=conv_id,
                role="user",
                content={"role": "user", "text": "landed late"},
            )
        )

    with caplog.at_level(logging.WARNING, logger="dudamel.compaction"):
        record = await compactor.maybe_compact(
            conv_id, history, dropped=3, turn_key="turn-1", dropped_tainted=False
        )

    assert record is None
    assert fp.calls == []  # no summarizer call against a span we can't locate
    assert await _summary_rows(db, conv_id) == []  # and no wrong watermark written
    assert "shifted under compaction" in caplog.text
    await db.dispose()


async def test_newest_returns_none_when_no_summary_exists(tmp_path: Path) -> None:
    compactor, fp, db, conv_id = await _make(tmp_path, [])
    assert await compactor.newest(conv_id) is None
    await db.dispose()


async def test_turn_cache_does_not_accumulate_across_turns(tmp_path: Path) -> None:
    """`_turn_cache` exists only to memoize repeat calls WITHIN one turn's
    iteration loop (a fresh turn_key each turn) -- across many turns on the
    same conversation it must stay at one entry, not grow without bound for
    the life of the process."""
    compactor, fp, db, conv_id = await _make(tmp_path, [fake_text(f"gist-{i}") for i in range(50)])
    history = await _seed_messages(db, conv_id, 5)
    for i in range(50):
        await compactor.maybe_compact(
            conv_id, history, dropped=3, turn_key=f"turn-{i}", dropped_tainted=False
        )
        assert len(compactor._turn_cache) == 1
    await db.dispose()


async def test_newest_returns_the_latest_row(tmp_path: Path) -> None:
    compactor, fp, db, conv_id = await _make(tmp_path, [])
    async with db.session() as s:
        s.add(Summary(conversation_id=conv_id, up_to_message_id=1, text="older", tainted=False))
    async with db.session() as s:
        s.add(Summary(conversation_id=conv_id, up_to_message_id=2, text="newer", tainted=True))
    record = await compactor.newest(conv_id)
    assert record is not None
    assert record.text == "newer"
    assert record.tainted is True
    await db.dispose()
