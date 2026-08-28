"""Short notes, searchable by substring."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import or_, select

from dudamel import App


class NotesSettings(BaseModel):
    recent_limit: int = 5


app = App("notes", description="Short notes, searchable by substring", settings=NotesSettings)


class Note(app.Model, table="entries"):
    title: str
    body: str
    created_at: datetime = app.now()


def _like_pattern(query: str) -> str:
    """LIKE-escape so a query of "%" matches a literal percent sign.

    The backslash is escaped FIRST; doing it after would double-escape the
    escapes this function just added. Wrapping in %...% happens after escaping,
    which is the other easy way to get this wrong.
    """
    for character in ("\\", "%", "_"):
        query = query.replace(character, f"\\{character}")
    return f"%{query}%"


@app.tool
async def add_note(title: str, body: str) -> str:
    """Save a note."""
    async with app.db() as session:
        session.add(Note(title=title, body=body))
    return f"Saved: {title}"


@app.tool(read_only=True, external=True)
async def search_notes(query: str) -> str:
    """Find notes whose title or body contains `query`.

    Plain substring matching, case-insensitive. There is no ranking and no
    stemming, and a multi-word query matches the whole phrase rather than the
    individual words.
    """
    pattern = _like_pattern(query)
    async with app.db() as session:
        rows = (
            (
                await session.execute(
                    select(Note)
                    .where(
                        or_(
                            Note.title.ilike(pattern, escape="\\"),
                            Note.body.ilike(pattern, escape="\\"),
                        )
                    )
                    .order_by(Note.created_at.desc(), Note.id.desc())
                )
            )
            .scalars()
            .all()
        )
    if not rows:
        return "No notes matched."
    return "\n".join(f"[{row.id}] {row.title}" for row in rows)


@app.tool(read_only=True, external=True)
async def read_note(note_id: int) -> str:
    """Read one note in full."""
    async with app.db() as session:
        row = await session.get(Note, note_id)
        if row is None:
            return f"No note with id {note_id}."
        return f"{row.title}\n\n{row.body}"


@app.tool(confirm=True)
async def delete_note(note_id: int, title: str) -> str:
    """Delete one note permanently. `title` must match the note's own title."""
    async with app.db() as session:
        row = await session.get(Note, note_id)
        if row is None:
            return f"No note with id {note_id}."
        if row.title != title:
            return f"Refused: note {note_id} is {row.title!r}, not {title!r}."
        await session.delete(row)
        return f"Deleted: {row.title}"


@app.widget(title="Recent notes", renderer="list")
async def recent() -> list[dict]:
    async with app.db() as session:
        rows = (
            (
                await session.execute(
                    select(Note)
                    .order_by(Note.created_at.desc(), Note.id.desc())
                    .limit(app.settings.recent_limit)
                )
            )
            .scalars()
            .all()
        )
    if not rows:
        return [{"title": "No notes yet.", "subtitle": "Ask to save one."}]
    # `created_at` is stored naive UTC; the date shown is the operator's, so a
    # note written a moment ago is not dated yesterday beside the other cards.
    return [
        {"title": row.title, "subtitle": app.in_timezone(row.created_at).date().isoformat()}
        for row in rows
    ]
