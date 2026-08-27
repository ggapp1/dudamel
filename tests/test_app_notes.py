"""`notes_app` comes from tests/conftest.py: bound database AND bound settings."""

import pytest
from sqlalchemy import func, select


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
    can paste an email in directly. 6a-2a's design named this exact case --
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
