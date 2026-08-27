from datetime import UTC, date, datetime

from sqlalchemy import select


async def test_add_task_persists_it_and_says_so(tasks_app):
    from dudamel.apps.tasks import Task, add_task

    message = await add_task("buy milk", due=date(2026, 9, 1))

    assert "buy milk" in message
    async with tasks_app.db() as session:
        rows = (await session.execute(select(Task))).scalars().all()
    assert len(rows) == 1
    assert rows[0].title == "buy milk"
    assert rows[0].done is False
    assert rows[0].due == date(2026, 9, 1)


async def test_list_tasks_hides_done_ones_unless_asked(tasks_app):
    from dudamel.apps.tasks import Task, add_task, list_tasks

    await add_task("open one")
    await add_task("finished one")
    async with tasks_app.db() as session:
        row = (await session.execute(select(Task).where(Task.title == "finished one"))).scalar_one()
        row.done = True

    default = await list_tasks()
    assert "open one" in default
    assert "finished one" not in default

    everything = await list_tasks(include_done=True)
    assert "open one" in everything
    assert "finished one" in everything


async def test_list_tasks_says_so_when_there_are_none(tasks_app):
    from dudamel.apps.tasks import list_tasks

    assert "No tasks" in await list_tasks()


async def _one_task_id(tasks_app, title: str) -> int:
    from dudamel.apps.tasks import Task, add_task

    await add_task(title)
    async with tasks_app.db() as session:
        return (await session.execute(select(Task.id).where(Task.title == title))).scalar_one()


async def test_complete_task_marks_it_done(tasks_app):
    """`"Done" in message` would also pass on "Already done" -- anchor the prefix."""
    from dudamel.apps.tasks import Task, complete_task

    task_id = await _one_task_id(tasks_app, "water plants")

    assert (await complete_task(task_id)).startswith("Done:")
    async with tasks_app.db() as session:
        assert (await session.get(Task, task_id)).done is True


async def test_completing_twice_is_harmless(tasks_app):
    """This is a row button behind a browser guard with no behavioural coverage,
    and the model can call it twice in one batch. It must be a no-op, not an error."""
    from dudamel.apps.tasks import complete_task

    task_id = await _one_task_id(tasks_app, "stretch")
    await complete_task(task_id)

    assert "Already done" in await complete_task(task_id)


async def test_completing_a_missing_task_reports_instead_of_raising(tasks_app):
    from dudamel.apps.tasks import complete_task

    assert "No task with id 999" in await complete_task(999)


def test_complete_task_is_the_cards_button(tasks_app):
    """The Done button on the card depends on this label existing."""
    assert tasks_app.tools["complete_task"].action == "Done"


async def test_delete_task_removes_the_row(tasks_app):
    from dudamel.apps.tasks import Task, delete_task

    task_id = await _one_task_id(tasks_app, "cancel gym")

    assert "Deleted" in await delete_task(task_id, "cancel gym")
    async with tasks_app.db() as session:
        assert await session.get(Task, task_id) is None


async def test_delete_task_refuses_on_a_title_mismatch(tasks_app):
    """The corroborating title is verified, not decorative.

    It exists so the confirm prompt is answerable -- `delete_task(task_id=7)`
    asks a human to approve an opaque integer. If the title were unchecked, the
    prompt would be theatre.
    """
    from dudamel.apps.tasks import Task, delete_task

    task_id = await _one_task_id(tasks_app, "cancel gym")

    assert "Refused" in await delete_task(task_id, "something else")
    async with tasks_app.db() as session:
        assert await session.get(Task, task_id) is not None


async def test_deleting_a_missing_task_reports_instead_of_raising(tasks_app):
    from dudamel.apps.tasks import delete_task

    assert "No task with id 999" in await delete_task(999, "whatever")


def test_delete_task_confirms_and_is_not_a_button(tasks_app):
    """confirm= is the only flag that still fires under taint_mode = "off", and
    an unlabelled tool is unreachable from POST /api/action/{tool} entirely."""
    tool = tasks_app.tools["delete_task"]
    assert tool.confirm is True
    assert tool.action is None


def test_tasks_is_registered_in_the_suite():
    from dudamel.apps import SUITE_APPS, suite_versions_dir
    from dudamel.apps.tasks import app

    entry = SUITE_APPS["tasks"]
    assert entry.module == "dudamel.apps.tasks"
    assert entry.name == app.name
    assert entry.extra is None and entry.requires == ()

    versions = suite_versions_dir(entry)
    assert versions.is_dir()
    assert list(versions.glob("[0-9]*.py")), "empty lane: migrate would report up to date"


async def test_today_card_lists_due_tasks_each_with_its_own_done_button(tasks_app):
    from datetime import timedelta

    from dudamel.apps.tasks import Task, add_task
    from dudamel.widgets import run_widget

    today = date.today()
    await add_task("due today", due=today)
    await add_task("undated")
    await add_task("far future", due=today + timedelta(days=400))
    await add_task("already done", due=today)
    async with tasks_app.db() as session:
        row = (await session.execute(select(Task).where(Task.title == "already done"))).scalar_one()
        row.done = True

    card = await run_widget(
        tasks_app.widgets["today"], {"complete_task": tasks_app.tools["complete_task"]}
    )

    titles = [item["title"] for item in card["data"]]
    assert "due today" in titles
    assert "undated" in titles, "an undated task is still a task"
    assert "far future" not in titles
    assert "already done" not in titles

    # Each row's button must carry ITS OWN id -- with one row, a truthiness
    # check cannot tell a correct wiring from every button pointing at row 1.
    async with tasks_app.db() as session:
        expected = {
            title: (await session.execute(select(Task.id).where(Task.title == title))).scalar_one()
            for title in ("due today", "undated")
        }
    for item in card["data"]:
        assert item["action"]["tool"] == "complete_task"
        assert item["action"]["label"] == "Done"
        assert item["action"]["args"]["task_id"] == expected[item["title"]]


async def test_today_card_has_an_empty_state(tasks_app):
    """A brand-new database is every user's first impression of the homescreen."""
    from dudamel.widgets import run_widget

    card = await run_widget(tasks_app.widgets["today"], {})

    assert [item["title"] for item in card["data"]] == ["Nothing due."]
    assert card["data"][0]["action"] is None


# Measured, not chosen by intuition. A floating "11:00 UTC" property does not
# discriminate: at 2026-08-27T11:00Z, UTC, Auckland and New York all read the
# same calendar date, so a naive datetime.now(UTC).date() passes it. Auckland is
# UTC+12 April-September and UTC+13 otherwise, so the property is DST-dependent.
#
#   instant              UTC     America/New_York   Pacific/Auckland
#   2026-01-16T02:00Z    01-16   01-15  <-- differs  01-16
#   2026-08-27T13:00Z    08-27   08-27              08-28  <-- differs
NY_EVENING = datetime(2026, 1, 16, 2, 0, tzinfo=UTC)  # 21:00 on 01-15 in UTC-5
AUCKLAND_TOMORROW = datetime(2026, 8, 27, 13, 0, tzinfo=UTC)


def _freeze(monkeypatch, module, instant: datetime) -> None:
    """Pin `datetime.now(UTC)` inside one app module."""

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz) if tz else instant.replace(tzinfo=None)

    monkeypatch.setattr(module, "datetime", _Frozen)


def test_local_today_is_the_users_day_not_the_servers(monkeypatch):
    """The 7.1 scenario verbatim: a habit or task acted on at 21:00 in UTC-5
    belongs to that evening, not to the UTC day that has already started."""
    from dudamel.apps import tasks

    _freeze(monkeypatch, tasks, NY_EVENING)
    assert tasks._local_today("America/New_York") == date(2026, 1, 15)  # naive UTC: 01-16
    assert tasks._local_today("UTC") == date(2026, 1, 16)

    _freeze(monkeypatch, tasks, AUCKLAND_TOMORROW)
    assert tasks._local_today("Pacific/Auckland") == date(2026, 8, 28)  # naive UTC: 08-27


async def test_the_card_routes_through_the_configured_timezone(tasks_app, monkeypatch):
    """A passing helper test says nothing about whether the widget calls it.

    Swap `_local_today(settings.timezone)` for `date.today()` in the widget body
    and the test above still passes; this one does not.
    """
    from dudamel.apps import tasks
    from dudamel.apps.tasks import add_task
    from dudamel.widgets import run_widget

    tasks_app.bind_settings({"horizon_days": 0, "timezone": "Pacific/Auckland"})
    _freeze(monkeypatch, tasks, AUCKLAND_TOMORROW)
    # Due on the Auckland day, which is NOT the UTC day at this instant.
    await add_task("local-day task", due=date(2026, 8, 28))

    card = await run_widget(
        tasks_app.widgets["today"], {"complete_task": tasks_app.tools["complete_task"]}
    )

    assert "local-day task" in [item["title"] for item in card["data"]]


async def test_horizon_days_is_a_boundary_not_a_vague_fence(tasks_app, monkeypatch):
    """A task due tomorrow is in at horizon_days=1 and out at 0.

    Fencing at "today" versus "today + 400 days" cannot detect an off-by-one, so
    the boundary is tested at the boundary.
    """
    from dudamel.apps import tasks
    from dudamel.apps.tasks import add_task
    from dudamel.widgets import run_widget

    _freeze(monkeypatch, tasks, AUCKLAND_TOMORROW)  # UTC day is 2026-08-27
    await add_task("tomorrow", due=date(2026, 8, 28))
    actions = {"complete_task": tasks_app.tools["complete_task"]}

    tasks_app.bind_settings({"horizon_days": 1})
    included = await run_widget(tasks_app.widgets["today"], actions)
    assert "tomorrow" in [item["title"] for item in included["data"]]

    tasks_app.bind_settings({"horizon_days": 0})
    excluded = await run_widget(tasks_app.widgets["today"], actions)
    assert "tomorrow" not in [item["title"] for item in excluded["data"]]


def test_settings_defaults_and_overrides():
    from dudamel.apps.tasks import TasksSettings

    assert TasksSettings().horizon_days == 1
    assert TasksSettings().timezone == "UTC"
    assert TasksSettings(timezone="Europe/Lisbon").horizon_days == 1


def test_an_unknown_timezone_is_refused_at_config_load_naming_the_app(tasks_app):
    """7.1 requires the message name the app, and that naming happens in
    `bind_settings` -- not in the model -- so this asserts through that path."""
    import pytest

    from dudamel.exceptions import AppSettingsError

    with pytest.raises(AppSettingsError, match="tasks"):
        tasks_app.bind_settings({"timezone": "Mars/Olympus"})


def test_an_unknown_settings_key_is_refused(tasks_app):
    """Alias-aware rejection, exercised against a real app for the first
    time: pydantic ignores extras by default, so this lives in bind_settings."""
    import pytest

    from dudamel.exceptions import AppSettingsError

    with pytest.raises(AppSettingsError):
        tasks_app.bind_settings({"horizon_dayz": 3})
