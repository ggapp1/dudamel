"""`habits_app` comes from tests/conftest.py: bound database AND bound settings."""

from datetime import date, timedelta
from zoneinfo import ZoneInfo

import pytest
from conftest import _freeze_app_clock
from sqlalchemy import func, select

from dudamel.exceptions import AppSettingsError

TODAY = date(2026, 8, 27)
# The same day, as the instant the clock is pinned to. `habits_app` binds UTC,
# so midday UTC reads as TODAY with hours of margin on either side.
TODAY_AT_MIDDAY = "2026-08-27T12:00:00Z"


# _streak is a pure function over a set of dates on purpose: streak logic is
# where off-by-ones live, and testing it without a database makes every edge
# case one cheap assertion.
def test_streak_counts_back_from_today_and_stops_at_a_gap():
    from dudamel.apps.habits import _streak

    assert _streak({TODAY, TODAY - timedelta(days=1), TODAY - timedelta(days=3)}, TODAY) == 2


def test_today_alone_is_a_streak_of_one():
    from dudamel.apps.habits import _streak

    assert _streak({TODAY}, TODAY) == 1


def test_not_having_ticked_yet_today_does_not_break_a_live_streak():
    """Yesterday anchors a live streak: a habit not yet done TODAY has not been
    broken, and showing 0 until the user ticks would be a lie every morning."""
    from dudamel.apps.habits import _streak

    assert _streak({TODAY - timedelta(days=1), TODAY - timedelta(days=2)}, TODAY) == 2


def test_a_streak_that_ended_before_yesterday_is_dead():
    """THE test. Without it, anchoring on max(days) instead of today/yesterday
    passes every other case here, and a month-dead habit reads as live forever.
    """
    from dudamel.apps.habits import _streak

    assert _streak({TODAY - timedelta(days=2)}, TODAY) == 0
    assert _streak({TODAY - timedelta(days=30), TODAY - timedelta(days=31)}, TODAY) == 0


def test_a_future_tick_does_not_inflate_the_streak():
    """Clock skew, or moving `timezone` eastward, can put tomorrow in the set."""
    from dudamel.apps.habits import _streak

    assert _streak({TODAY + timedelta(days=1), TODAY, TODAY - timedelta(days=1)}, TODAY) == 2
    assert _streak({TODAY + timedelta(days=1)}, TODAY) == 0


def test_streak_of_nothing_is_zero():
    from dudamel.apps.habits import _streak

    assert _streak(set(), TODAY) == 0


async def _habit_id(habits_app, name: str) -> int:
    from dudamel.apps.habits import Habit, add_habit

    await add_habit(name)
    async with habits_app.db() as session:
        return (await session.execute(select(Habit.id).where(Habit.name == name))).scalar_one()


async def _tick_count(habits_app) -> int:
    from dudamel.apps.habits import Tick

    async with habits_app.db() as session:
        return (await session.execute(select(func.count()).select_from(Tick))).scalar_one()


async def test_ticking_twice_in_one_day_leaves_one_row(habits_app):
    """The dashboard's no-re-submit guard has no behavioural coverage, and the
    model can call this twice in one batch. It must be a no-op either way."""
    from dudamel.apps.habits import tick_habit

    habit_id = await _habit_id(habits_app, "floss")

    first = await tick_habit(habit_id)
    second = await tick_habit(habit_id)

    assert first.startswith("Ticked:")
    assert "already" in second.lower()
    assert await _tick_count(habits_app) == 1


async def test_the_schema_itself_rejects_a_duplicate_tick(habits_app):
    """The guard above is a read-then-write race; this constraint is what makes
    a double-click harmless when the guard loses. Inserted out-of-band, because
    tick_habit's own guard would otherwise be what the test measures."""
    from sqlalchemy.exc import IntegrityError

    from dudamel.apps.habits import Tick

    habit_id = await _habit_id(habits_app, "floss")
    async with habits_app.db() as session:
        session.add(Tick(habit_id=habit_id, day=TODAY))

    with pytest.raises(IntegrityError):
        async with habits_app.db() as session:
            session.add(Tick(habit_id=habit_id, day=TODAY))


async def test_tick_reports_already_for_a_row_it_did_not_write(habits_app, monkeypatch):
    """The path a concurrent request actually hits: the row is there, but this
    call did not put it there."""
    from dudamel.apps.habits import Tick, tick_habit

    habit_id = await _habit_id(habits_app, "floss")
    _freeze_app_clock(monkeypatch, TODAY_AT_MIDDAY)
    async with habits_app.db() as session:
        session.add(Tick(habit_id=habit_id, day=TODAY))

    assert "already" in (await tick_habit(habit_id)).lower()
    assert await _tick_count(habits_app) == 1


async def test_ticking_a_missing_habit_reports_instead_of_raising(habits_app):
    from dudamel.apps.habits import tick_habit

    assert "No habit with id 999" in await tick_habit(999)


async def test_undo_removes_only_todays_tick(habits_app, monkeypatch):
    """Drop the day filter from the delete and this habit's entire history goes
    with one tap of Undo -- unrecoverably, since the streak IS those rows."""
    from dudamel.apps.habits import Tick, untick_habit

    habit_id = await _habit_id(habits_app, "read")
    _freeze_app_clock(monkeypatch, TODAY_AT_MIDDAY)
    async with habits_app.db() as session:
        session.add(Tick(habit_id=habit_id, day=TODAY))
        session.add(Tick(habit_id=habit_id, day=TODAY - timedelta(days=1)))

    assert "Unticked" in await untick_habit(habit_id)

    async with habits_app.db() as session:
        remaining = (await session.execute(select(Tick.day))).scalars().all()
    assert remaining == [TODAY - timedelta(days=1)], "yesterday must survive"


async def test_undo_when_not_ticked_reports_and_deletes_nothing(habits_app, monkeypatch):
    from dudamel.apps.habits import Tick, untick_habit

    habit_id = await _habit_id(habits_app, "read")
    _freeze_app_clock(monkeypatch, TODAY_AT_MIDDAY)
    async with habits_app.db() as session:
        session.add(Tick(habit_id=habit_id, day=TODAY - timedelta(days=1)))

    assert "Not ticked today" in await untick_habit(habit_id)
    assert await _tick_count(habits_app) == 1


async def test_undo_on_a_missing_habit_reports_instead_of_raising(habits_app):
    from dudamel.apps.habits import untick_habit

    assert "No habit with id 999" in await untick_habit(999)


async def test_tick_undo_tick_returns_to_one_row(habits_app):
    from dudamel.apps.habits import tick_habit, untick_habit

    habit_id = await _habit_id(habits_app, "walk")

    await tick_habit(habit_id)
    await untick_habit(habit_id)
    await tick_habit(habit_id)

    assert await _tick_count(habits_app) == 1


async def test_list_habits_reports_each_streak_independently(habits_app, monkeypatch):
    """Group the ticks by the wrong key and every habit shares one set."""
    from dudamel.apps.habits import Tick, list_habits

    long_run = await _habit_id(habits_app, "meditate")
    await _habit_id(habits_app, "untouched")
    _freeze_app_clock(monkeypatch, TODAY_AT_MIDDAY)
    async with habits_app.db() as session:
        for delta in (0, 1, 2):
            session.add(Tick(habit_id=long_run, day=TODAY - timedelta(days=delta)))

    listing = await list_habits()

    assert "meditate — 3 day streak" in listing
    assert "untouched — 0 day streak" in listing


async def test_list_habits_says_so_when_there_are_none(habits_app):
    from dudamel.apps.habits import list_habits

    assert "No habits yet" in await list_habits()


async def test_the_tick_day_is_the_users_day(habits_app, monkeypatch):
    """A tick at 21:00 in UTC-5 belongs to that evening, not to the UTC day that
    has already begun. 2026-01-16T02:00Z is exactly that instant.

    A passing test of the framework's own date says nothing about whether the
    tool calls it. Swap `app.today()` for `date.today()` in `tick_habit` and
    this is the test that notices.
    """
    from dudamel.apps.habits import Tick, tick_habit

    habits_app.bind_timezone(ZoneInfo("America/New_York"))
    _freeze_app_clock(monkeypatch, "2026-01-16T02:00:00Z")
    habit_id = await _habit_id(habits_app, "journal")
    await tick_habit(habit_id)

    async with habits_app.db() as session:
        day = (await session.execute(select(Tick.day))).scalar_one()
    assert day == date(2026, 1, 15), "naive UTC would record 2026-01-16"


def _actions(habits_app) -> dict:
    return {
        "tick_habit": habits_app.tools["tick_habit"],
        "untick_habit": habits_app.tools["untick_habit"],
    }


async def test_today_card_offers_tick_then_undo_per_habit(habits_app, monkeypatch):
    """Two habits in opposite states: computing `done` globally rather than per
    row would pass with one habit and fails here."""
    from dudamel.apps.habits import Tick
    from dudamel.widgets import run_widget

    ticked = await _habit_id(habits_app, "ticked one")
    await _habit_id(habits_app, "untouched one")
    _freeze_app_clock(monkeypatch, TODAY_AT_MIDDAY)
    async with habits_app.db() as session:
        session.add(Tick(habit_id=ticked, day=TODAY))

    card = await run_widget(habits_app.widgets["today"], _actions(habits_app))
    rows = {item["title"]: item["action"] for item in card["data"]}

    assert rows["ticked one"]["tool"] == "untick_habit"
    assert rows["ticked one"]["label"] == "Undo"
    assert rows["untouched one"]["tool"] == "tick_habit"
    assert rows["untouched one"]["label"] == "Tick"


async def test_today_card_states_the_streak_exactly(habits_app, monkeypatch):
    """`"1" in subtitle` would pass for 1, 10, 11, 21, a hardcoded streak, and a
    date containing a 1. Assert the whole string."""
    from dudamel.apps.habits import Tick
    from dudamel.widgets import run_widget

    habit_id = await _habit_id(habits_app, "walk")
    _freeze_app_clock(monkeypatch, TODAY_AT_MIDDAY)
    async with habits_app.db() as session:
        for delta in (0, 1, 2):
            session.add(Tick(habit_id=habit_id, day=TODAY - timedelta(days=delta)))

    card = await run_widget(habits_app.widgets["today"], _actions(habits_app))

    assert card["data"][0]["subtitle"] == "3 day streak · done today"


async def test_today_card_marks_an_untimely_habit_as_not_done(habits_app, monkeypatch):
    from dudamel.apps.habits import Tick
    from dudamel.widgets import run_widget

    habit_id = await _habit_id(habits_app, "walk")
    _freeze_app_clock(monkeypatch, TODAY_AT_MIDDAY)
    async with habits_app.db() as session:
        session.add(Tick(habit_id=habit_id, day=TODAY - timedelta(days=1)))

    card = await run_widget(habits_app.widgets["today"], _actions(habits_app))

    assert card["data"][0]["subtitle"] == "1 day streak"


async def test_today_card_empty_state(habits_app):
    from dudamel.widgets import run_widget

    card = await run_widget(habits_app.widgets["today"], {})

    assert [item["title"] for item in card["data"]] == ["No habits yet."]
    assert card["data"][0]["action"] is None


def test_a_stale_habits_timezone_is_rejected_by_binding():
    """The behaviour, not its spelling. `settings_model is None` is satisfied by
    an empty settings model too, and both reject the key -- so that assertion
    cannot tell which outcome it pinned."""
    from dudamel.apps.habits import app as habits_app

    with pytest.raises(AppSettingsError, match="timezone"):
        habits_app.bind_settings({"timezone": "UTC"})
