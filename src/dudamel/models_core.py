"""Framework-owned tables. Schema changes here REQUIRE a new migration in
src/dudamel/migrations/versions/ — never edit 0001 after release."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
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
    result_preview: Mapped[str | None] = mapped_column(Text, nullable=True)
    # reserved; never populated
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
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
