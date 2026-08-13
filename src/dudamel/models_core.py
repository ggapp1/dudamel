"""Framework-owned tables. Schema changes here REQUIRE a new migration in
src/dudamel/migrations/versions/ — never edit 0001 after release."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class CoreBase(DeclarativeBase):
    pass


class Conversation(CoreBase):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel: Mapped[str] = mapped_column(String(255), unique=True)  # e.g. "telegram:12345"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Message(CoreBase):
    __tablename__ = "messages"
    __table_args__ = (
        # DB-level backstop for append() dedupe (see migration 0003): NULL
        # client_msg_ids never conflict with each other or anything else in
        # a SQLite/Postgres unique index, so this only constrains rows that
        # actually carry a client_msg_id.
        Index("uq_messages_conv_client_msg", "conversation_id", "client_msg_id", unique=True),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(32))  # user|assistant|tool
    content: Mapped[dict] = mapped_column(JSON)  # provider-neutral message body
    client_msg_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Activity(CoreBase):
    __tablename__ = "activity"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True
    )
    tool: Mapped[str] = mapped_column(String(128))
    args: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32))  # ok|error|declined|confirmed
    # Who invoked this and through which surface. Nullable because rows
    # written before these columns existed cannot be attributed, and
    # inventing a value for them would be a lie in an audit log.
    actor: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str | None] = mapped_column(String(16), nullable=True)  # router|web|telegram
    result_preview: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Token/cost accounting lives in llm_calls (per call, with a
    # conversation_id), never here -- 0001's reserved tokens_in/tokens_out/
    # cost_usd columns were dropped in migration 0005.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class JobRun(CoreBase):
    __tablename__ = "job_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(255), index=True)
    status: Mapped[str] = mapped_column(String(32))  # ok|error|timeout|skipped|misfired|cancelled
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)


class PendingConfirmation(CoreBase):
    __tablename__ = "pending_confirmations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # uuid4
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(255))
    tool: Mapped[str] = mapped_column(String(128))
    args: Mapped[dict] = mapped_column(JSON)
    loop_state: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime)


class LlmCall(CoreBase):
    __tablename__ = "llm_calls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tier: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(128))
    tokens_in: Mapped[int] = mapped_column(Integer)
    tokens_out: Mapped[int] = mapped_column(Integer)
    conversation_id: Mapped[int | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)


class Summary(CoreBase):
    __tablename__ = "summaries"
    __table_args__ = (
        # Lets a query scan a conversation's summaries newest-first without a
        # separate sort step -- `newest()` and the pruning-on-write query both
        # do exactly that. Its leading column also serves every plain
        # conversation_id lookup, which is why the column carries no index of
        # its own (0004's was dropped in 0005).
        Index("ix_summaries_conversation_id_id", "conversation_id", "id"),
        # One summary per watermark per conversation: the reuse check in
        # compaction.py relies on there being no more than one row for a
        # given (conversation_id, up_to_message_id) pair.
        Index("uq_summaries_conv_upto", "conversation_id", "up_to_message_id", unique=True),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    # Watermark: the id of the newest Message row this summary covers. A
    # window whose dropped span ends at or before this id can reuse the
    # summary instead of calling the model again.
    up_to_message_id: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    # Computed from the provenance of the summarized rows (any MCP-origin
    # tool call in the dropped span), never from the summarizer's own
    # output -- see compaction.py.
    tainted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
