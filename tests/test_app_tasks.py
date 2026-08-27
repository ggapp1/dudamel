from datetime import date

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
