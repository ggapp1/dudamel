from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

IN_DB_SCOPE: ContextVar[bool] = ContextVar("in_db_scope", default=False)


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        self.engine = create_async_engine(url)
        if url.startswith("sqlite"):

            @event.listens_for(self.engine.sync_engine, "connect")
            def _sqlite_pragmas(dbapi_conn: Any, _record: Any) -> None:
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.close()

        self._factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def dispose(self) -> None:
        """Dispose the underlying engine's connection pool."""
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        token = IN_DB_SCOPE.set(True)
        try:
            async with self._factory() as s:
                try:
                    yield s
                    await s.commit()
                except BaseException:
                    await s.rollback()
                    raise
        finally:
            IN_DB_SCOPE.reset(token)
