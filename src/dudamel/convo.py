from __future__ import annotations

from sqlalchemy import select

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
            row = Conversation(channel=channel)
            s.add(row)
            await s.flush()  # assign pk before the context manager commits
            return row.id

    async def append(
        self,
        conversation_id: int,
        message: Message,
        *,
        client_msg_id: str | None = None,
    ) -> bool:
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
        return True

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
