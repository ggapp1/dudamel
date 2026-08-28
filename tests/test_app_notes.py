"""`notes_app` comes from tests/conftest.py: bound database AND bound settings."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from conftest import _freeze_app_clock
from sqlalchemy import func, select

from dudamel import App
from dudamel.exceptions import RuntimeNotBoundError


async def _note_id(notes_app, title: str, body: str = "body") -> int:
    from dudamel.apps.notes import Note, add_note

    await add_note(title, body)
    async with notes_app.db() as session:
        return (await session.execute(select(Note.id).where(Note.title == title))).scalar_one()


async def test_add_and_read_a_note(notes_app):
    from dudamel.apps.notes import read_note

    note_id = await _note_id(notes_app, "groceries", "milk, eggs")

    assert "milk, eggs" in await read_note(note_id)


async def test_search_matches_title_and_body_case_insensitively(notes_app):
    from dudamel.apps.notes import add_note, search_notes

    await add_note("Groceries", "milk and EGGS")
    await add_note("unrelated", "nothing here")

    assert "Groceries" in await search_notes("grocer")
    assert "Groceries" in await search_notes("eggs")
    assert "unrelated" not in await search_notes("eggs")


async def test_search_says_so_when_nothing_matches(notes_app):
    from dudamel.apps.notes import search_notes

    assert "No notes" in await search_notes("nothing matches this")


@pytest.mark.parametrize("wildcard", ["%", "_"])
async def test_search_treats_sql_wildcards_as_literal_text(notes_app, wildcard):
    """A naive f"%{query}%" makes these wildcards: "%" would dump the whole
    corpus and "_" would match any single character."""
    from dudamel.apps.notes import search_notes

    await _note_id(notes_app, "real note")

    assert "real note" not in await search_notes(wildcard)


async def test_search_still_finds_a_literal_percent(notes_app):
    """The other direction: over-escaping breaks honest queries."""
    from dudamel.apps.notes import search_notes

    await _note_id(notes_app, "battery", "charged to 100% today")

    assert "battery" in await search_notes("100%")


async def test_reading_a_missing_note_reports_instead_of_raising(notes_app):
    from dudamel.apps.notes import read_note

    assert "No note with id 999" in await read_note(999)


def test_the_note_readers_taint_the_turn(notes_app):
    """Self-contained answers the FETCH direction only.

    What a note *contains* may be attacker text: the model composes add_note's
    arguments after reading its window, which can hold MCP results, and a user
    can paste an email in directly. This is the known limitation that
    stored external content is untainted "unless the reader tool is marked too".
    Asserted through `untrusted`, the single taint predicate, so it still holds
    if that predicate widens.
    """
    assert notes_app.tools["search_notes"].untrusted is True
    assert notes_app.tools["read_note"].untrusted is True


def test_delete_note_confirms_and_is_not_a_button(notes_app):
    tool = notes_app.tools["delete_note"]
    assert tool.confirm is True
    assert tool.action is None


async def test_delete_note_removes_the_row(notes_app):
    from dudamel.apps.notes import Note, delete_note

    note_id = await _note_id(notes_app, "throwaway")

    assert "Deleted" in await delete_note(note_id, "throwaway")
    async with notes_app.db() as session:
        assert await session.get(Note, note_id) is None


async def test_delete_note_refuses_on_a_title_mismatch(notes_app):
    from dudamel.apps.notes import Note, delete_note

    note_id = await _note_id(notes_app, "tax docs")

    assert "Refused" in await delete_note(note_id, "shopping list")
    async with notes_app.db() as session:
        assert await session.get(Note, note_id) is not None


async def test_deleting_a_missing_note_reports_instead_of_raising(notes_app):
    from dudamel.apps.notes import delete_note

    assert "No note with id 999" in await delete_note(999, "whatever")


async def test_recent_card_shows_the_newest_first(notes_app):
    """ "Recent" is the card's entire claim. Asserting only the LENGTH would pass
    on an implementation showing the five OLDEST notes."""
    from dudamel.widgets import run_widget

    for index in range(8):
        await _note_id(notes_app, f"note {index}")

    card = await run_widget(notes_app.widgets["recent"], {})

    assert [item["title"] for item in card["data"]] == [f"note {i}" for i in (7, 6, 5, 4, 3)]


async def test_recent_card_carries_no_actions(notes_app):
    """Notes is an archive app: you query it, you do not tap it. Shipping one
    read-only card is deliberate documentation that this is a legitimate shape,
    and it keeps a destructive delete off a surface where one tap is consent."""
    from dudamel.widgets import run_widget

    await _note_id(notes_app, "only note")

    card = await run_widget(notes_app.widgets["recent"], {})

    # Pin the row first: `all(...)` over an empty list is vacuously true.
    assert [item["title"] for item in card["data"]] == ["only note"]
    assert all(item["action"] is None for item in card["data"])
    assert card["actions"] == []


async def test_recent_card_empty_state(notes_app):
    from dudamel.widgets import run_widget

    card = await run_widget(notes_app.widgets["recent"], {})

    assert [item["title"] for item in card["data"]] == ["No notes yet."]


async def test_recent_limit_is_configurable(notes_app):
    from dudamel.widgets import run_widget

    for index in range(8):
        await _note_id(notes_app, f"note {index}")
    notes_app.bind_settings({"recent_limit": 2})

    card = await run_widget(notes_app.widgets["recent"], {})

    assert len(card["data"]) == 2


async def test_a_note_count_sanity(notes_app):
    from dudamel.apps.notes import Note

    await _note_id(notes_app, "one")
    async with notes_app.db() as session:
        assert (await session.execute(select(func.count()).select_from(Note))).scalar_one() == 1


# 11:30 UTC on this date is 00:30 the NEXT day in Auckland (NZDT, +13:00), so a
# date rendered from the stored naive-UTC value differs from the operator's.
WRITTEN_AT_UTC = "2026-01-16T11:30:00Z"
NAIVE_UTC_STAMP = datetime(2026, 1, 16, 11, 30)
AUCKLAND_LOCAL_DATE = "2026-01-17"


async def test_recent_card_dates_a_note_in_the_operators_zone(notes_app, monkeypatch):
    """The note you just wrote must not read as yesterday's.

    `created_at` is stored naive UTC, so rendering it straight puts the card a
    day behind every other surface for an operator east of UTC -- the homescreen
    would show a habit ticked for one date and a note written the same minute
    dated the day before.
    """
    from dudamel.apps.notes import Note
    from dudamel.widgets import run_widget

    notes_app.bind_timezone(ZoneInfo("Pacific/Auckland"))
    _freeze_app_clock(monkeypatch, WRITTEN_AT_UTC)
    async with notes_app.db() as session:
        session.add(Note(title="just written", body="body", created_at=NAIVE_UTC_STAMP))

    card = await run_widget(notes_app.widgets["recent"], {})

    assert [item["subtitle"] for item in card["data"]] == [AUCKLAND_LOCAL_DATE]
    # The same date the rest of the homescreen is showing at this instant.
    assert card["data"][0]["subtitle"] == notes_app.today().isoformat()


def test_a_stored_timestamp_reads_as_an_aware_local_time() -> None:
    """The wall clock is asserted alongside the instant on purpose.

    Comparing two aware datetimes compares instants, and the naive-value bug
    this seam exists to prevent preserves the instant on a UTC host -- so an
    equality check alone passes there and proves nothing.
    """
    app = App("n", description="d")
    app.bind_timezone(ZoneInfo("Pacific/Auckland"))

    converted = app.in_timezone(NAIVE_UTC_STAMP)

    assert converted.replace(tzinfo=None) == datetime(2026, 1, 17, 0, 30)
    assert converted.utcoffset() == timedelta(hours=13)
    assert converted == NAIVE_UTC_STAMP.replace(tzinfo=UTC)


def test_the_timezone_seams_refuse_to_answer_before_a_zone_is_bound() -> None:
    """Same failure `today()` and `db()` give: loud at the line that forgot to
    bind, rather than a plausible wrong date computed from the host."""
    app = App("n", description="d")

    with pytest.raises(RuntimeNotBoundError):
        _ = app.timezone
    with pytest.raises(RuntimeNotBoundError):
        app.in_timezone(NAIVE_UTC_STAMP)
