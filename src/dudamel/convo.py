from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from dudamel.db import Database
from dudamel.llm.types import Message
from dudamel.models_core import Conversation
from dudamel.models_core import Message as MessageRow


class ConversationStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def get_or_create(self, channel: str) -> int:
        async with self._db.session() as s:
            existing = (
                await s.execute(select(Conversation).where(Conversation.channel == channel))
            ).scalar_one_or_none()
            if existing is not None:
                return existing.id
        try:
            async with self._db.session() as s:
                row = Conversation(channel=channel)
                s.add(row)
                await s.flush()  # assign pk before the context manager commits
                return row.id
        except IntegrityError:
            # Lost a race with a concurrent first-touch of the same new
            # channel: the unique constraint on `channel` rejected our
            # insert. Re-select in a fresh session (the prior one is dead
            # after rollback) and return the winner's id instead of
            # propagating a raw IntegrityError to the caller.
            async with self._db.session() as s:
                return (
                    (await s.execute(select(Conversation).where(Conversation.channel == channel)))
                    .scalar_one()
                    .id
                )

    async def append(
        self,
        conversation_id: int,
        message: Message,
        *,
        client_msg_id: str | None = None,
    ) -> bool:
        try:
            async with self._db.session() as s:
                if client_msg_id is not None:
                    dup = (
                        await s.execute(
                            select(MessageRow.id).where(
                                MessageRow.conversation_id == conversation_id,
                                MessageRow.client_msg_id == client_msg_id,
                            )
                        )
                    ).first()
                    if dup is not None:
                        return False
                s.add(
                    MessageRow(
                        conversation_id=conversation_id,
                        role=message.role,
                        content=message.to_dict(),
                        client_msg_id=client_msg_id,
                    )
                )
        except IntegrityError:
            # Lost a race with a concurrent append carrying the same
            # (conversation_id, client_msg_id): the DB-level unique index
            # (see migration 0003) rejected our insert on commit. The
            # pre-check above is a fast path, not a guarantee -- this is
            # the backstop that makes dedupe airtight under races.
            return False
        return True

    async def append_many(self, conversation_id: int, messages: list[Message]) -> None:
        """Append several messages in ONE transaction.

        Exists for the assistant-plus-tool-results group: appended one at a
        time, a crash between the assistant's tool_calls message and its
        results leaves a persisted tool_call nothing ever answers. One
        transaction makes the group all-or-nothing.

        Deliberately does NOT take a `client_msg_id`: only the single-message
        `append` above is ever handed one (the inbound user message), so
        there is no dedupe path here to get subtly wrong. Equally deliberately,
        this RAISES rather than returning a bool -- `append`'s blanket
        `except IntegrityError: return False` is a dedupe signal, and every
        caller of this method ignores return values, so swallowing a genuine
        integrity failure here would lose messages in silence.

        Each message is flushed individually so that at most one row is ever
        pending. That keeps ordering independent of how a given backend and
        driver version choose to emit a batch: SQLAlchemy's insertmanyvalues
        path can fold several pending rows into one multi-row INSERT ...
        RETURNING, and neither SQLite nor Postgres documents the order in
        which such a statement assigns primary keys. `recent()` orders by id,
        so a group whose tool results outranked the assistant message that
        called them would read back scrambled -- and a history like that is
        rejected outright by provider APIs. One row per flush sidesteps the
        question rather than depending on any backend's current behaviour.
        """
        if not messages:
            return
        async with self._db.session() as s:
            for message in messages:
                s.add(
                    MessageRow(
                        conversation_id=conversation_id,
                        role=message.role,
                        content=message.to_dict(),
                    )
                )
                await s.flush()

    async def recent(self, conversation_id: int, limit: int = 200) -> list[Message]:
        async with self._db.session() as s:
            rows = (
                (
                    await s.execute(
                        select(MessageRow)
                        .where(MessageRow.conversation_id == conversation_id)
                        .order_by(MessageRow.id.desc())
                        .limit(limit)
                    )
                )
                .scalars()
                .all()
            )
        return [Message.from_dict(r.content) for r in reversed(rows)]
