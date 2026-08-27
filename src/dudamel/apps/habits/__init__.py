"""Daily habits and streaks."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, field_validator
from sqlalchemy import UniqueConstraint, select
from sqlalchemy.exc import IntegrityError

from dudamel import App


class HabitsSettings(BaseModel):
    # The framework has no timezone of its own; a streak is meaningless without
    # one -- a tick at 21:00 in UTC-5 recorded on
    # the next UTC day makes a perfect streak read as broken.
    timezone: str = "UTC"

    @field_validator("timezone")
    @classmethod
    def _known_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value


app = App("habits", description="Daily habits and streaks", settings=HabitsSettings)


class Habit(app.Model, table="habits"):
    name: str
    created_at: datetime = app.now()


class Tick(app.Model, table="ticks"):
    habit_id: int
    # A LOCAL date -- the only temporal column in this system that is not UTC.
    # Every datetime here is naive-UTC by convention, but the day a tick belongs
    # to is the user's day, and storing it as a UTC date would reintroduce the
    # exact off-by-one the timezone setting exists to fix. It is a `date` rather
    # than a `datetime` precisely because a date has no timezone left to be
    # wrong about once it has been resolved.
    day: date

    __table_args__ = (UniqueConstraint("habit_id", "day", name="uq_habits_ticks_habit_day"),)


def _local_today(tz: str) -> date:
    """Today in `tz`. Not `date.today()`: the server's clock is not the user's."""
    return datetime.now(UTC).astimezone(ZoneInfo(tz)).date()


def _streak(days: set[date], today: date) -> int:
    """Consecutive days ending today, or yesterday if today is not ticked yet.

    Immune to DST by construction: this is `timedelta(days=1)` arithmetic over
    `date` objects, which is calendar-day arithmetic with no offsets involved.
    Do not "fix" it to use datetimes.
    """
    if not days:
        return 0
    # Anchoring on today-or-yesterday, NOT on max(days): a habit last done a
    # month ago has a streak of zero, however tidy its old run was.
    cursor = today if today in days else today - timedelta(days=1)
    count = 0
    while cursor in days:
        count += 1
        cursor -= timedelta(days=1)
    return count


@app.tool
async def add_habit(name: str) -> str:
    """Start tracking a daily habit."""
    async with app.db() as session:
        session.add(Habit(name=name))
    return f"Tracking: {name}"


async def _tick_for(session, habit_id: int, day: date) -> Tick | None:
    result = await session.execute(select(Tick).where(Tick.habit_id == habit_id, Tick.day == day))
    return result.scalar_one_or_none()


@app.tool(action="Tick")
async def tick_habit(habit_id: int) -> str:
    """Mark a habit as done for today."""
    async with app.db() as session:
        habit = await session.get(Habit, habit_id)
        if habit is None:
            return f"No habit with id {habit_id}."
        day = _local_today(app.settings.timezone)
        if await _tick_for(session, habit_id, day) is not None:
            return f"Already ticked today: {habit.name}"
        session.add(Tick(habit_id=habit_id, day=day))
        try:
            # Flushing here, rather than letting the commit raise on the way
            # out, is what makes the race recoverable: the read above and this
            # insert are not atomic, so a concurrent tick can land between them.
            # The unique constraint is the backstop, and catching it turns a
            # lost race into the same answer the guard would have given.
            await session.flush()
        except IntegrityError:
            await session.rollback()
            return f"Already ticked today: {habit.name}"
        return f"Ticked: {habit.name}"


@app.tool(action="Undo")
async def untick_habit(habit_id: int) -> str:
    """Undo today's tick for a habit."""
    async with app.db() as session:
        habit = await session.get(Habit, habit_id)
        if habit is None:
            return f"No habit with id {habit_id}."
        # Scoped to today deliberately: this is a one-tap button, and a delete
        # that ignored the day would take the whole history with it.
        tick = await _tick_for(session, habit_id, _local_today(app.settings.timezone))
        if tick is None:
            return f"Not ticked today: {habit.name}"
        await session.delete(tick)
        return f"Unticked: {habit.name}"


@app.tool(read_only=True)
async def list_habits() -> str:
    """List habits with their current streaks."""
    today = _local_today(app.settings.timezone)
    async with app.db() as session:
        habits = (
            (await session.execute(select(Habit).order_by(Habit.created_at, Habit.id)))
            .scalars()
            .all()
        )
        ticks = (await session.execute(select(Tick.habit_id, Tick.day))).all()
    if not habits:
        return "No habits yet."
    by_habit: dict[int, set[date]] = {}
    for tick_habit_id, day in ticks:
        by_habit.setdefault(tick_habit_id, set()).add(day)
    return "\n".join(
        f"[{habit.id}] {habit.name} — {_streak(by_habit.get(habit.id, set()), today)} day streak"
        for habit in habits
    )


@app.widget(title="Habits", renderer="list")
async def today() -> list[dict]:
    day = _local_today(app.settings.timezone)
    async with app.db() as session:
        habits = (
            (await session.execute(select(Habit).order_by(Habit.created_at, Habit.id)))
            .scalars()
            .all()
        )
        ticks = (await session.execute(select(Tick.habit_id, Tick.day))).all()
    if not habits:
        return [{"title": "No habits yet.", "subtitle": "Ask to start tracking one."}]
    by_habit: dict[int, set[date]] = {}
    for tick_habit_id, tick_day in ticks:
        by_habit.setdefault(tick_habit_id, set()).add(tick_day)
    items = []
    for habit in habits:
        days = by_habit.get(habit.id, set())
        done = day in days
        subtitle = f"{_streak(days, day)} day streak" + (" · done today" if done else "")
        items.append(
            {
                "title": habit.name,
                "subtitle": subtitle,
                # Resolved per row, so the card tells the truth after an HTMX
                # swap instead of offering an action that is now a no-op. The
                # labels come from each tool's own `action=`; no per-row
                # override is needed.
                "action": {
                    "tool": "untick_habit" if done else "tick_habit",
                    "args": {"habit_id": habit.id},
                },
            }
        )
    return items
