"""A to-do list: add tasks, see what is due, tick them off."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import select

from dudamel import App

app = App("tasks", description="A to-do list with due dates")


class Task(app.Model, table="items"):
    title: str
    done: bool = False
    due: date | None = None
    created_at: datetime = app.now()


@app.tool
async def add_task(title: str, due: date | None = None) -> str:
    """Add a task. `due` is an optional ISO date (YYYY-MM-DD)."""
    async with app.db() as session:
        session.add(Task(title=title, due=due))
    return f"Added: {title}" + (f" (due {due})" if due else "")


@app.tool(read_only=True)
async def list_tasks(include_done: bool = False) -> str:
    """List tasks, oldest first. Set include_done to also show finished ones."""
    async with app.db() as session:
        statement = select(Task).order_by(Task.created_at, Task.id)
        if not include_done:
            statement = statement.where(Task.done.is_(False))
        rows = (await session.execute(statement)).scalars().all()
    if not rows:
        return "No tasks."
    return "\n".join(
        f"[{row.id}] {'x' if row.done else ' '} {row.title}"
        + (f" (due {row.due})" if row.due else "")
        for row in rows
    )


@app.tool(action="Done")
async def complete_task(task_id: int) -> str:
    """Mark one task as done."""
    async with app.db() as session:
        row = await session.get(Task, task_id)
        if row is None:
            return f"No task with id {task_id}."
        if row.done:
            return f"Already done: {row.title}"
        row.done = True
        return f"Done: {row.title}"


@app.tool(confirm=True)
async def delete_task(task_id: int, title: str) -> str:
    """Delete one task permanently. `title` must match the task's own title.

    The second argument is not redundant. The confirm prompt renders the raw
    call, so `delete_task(task_id=7)` asks a person to approve an opaque
    integer; `delete_task(task_id=7, title='pay rent')` asks a question they can
    actually answer. It is verified below, so it cannot decay into decoration.
    """
    async with app.db() as session:
        row = await session.get(Task, task_id)
        if row is None:
            return f"No task with id {task_id}."
        if row.title != title:
            return f"Refused: task {task_id} is {row.title!r}, not {title!r}."
        await session.delete(row)
        return f"Deleted: {row.title}"
