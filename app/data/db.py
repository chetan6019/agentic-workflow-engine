"""Async SQLAlchemy engine and session factory."""

from __future__ import annotations

import asyncio

import structlog
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from contextlib import asynccontextmanager

from app.config import get_settings

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Return the cached async SQLAlchemy engine (asyncpg driver)."""
    url = get_settings().database_url
    return create_async_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)


def _session_factory() -> async_sessionmaker[AsyncSession]:
    """Build a sessionmaker bound to the shared engine."""
    return async_sessionmaker(get_engine(), expire_on_commit=False)


@asynccontextmanager
async def get_async_session():
    """Yield an AsyncSession; auto-commits on clean exit, rolls back on error."""
    factory = _session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Lightweight idempotent migrations for columns added after a table first shipped
# (create_all never alters existing tables).
_MIGRATIONS = [
    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS title VARCHAR(200)",
]


async def init_db(*, retries: int = 10, delay: float = 2.0) -> None:
    """Create all ORM tables, retrying if Postgres isn't ready yet."""
    from sqlalchemy import text

    from app.data.models import Base
    for attempt in range(1, retries + 1):
        try:
            async with get_engine().begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                for stmt in _MIGRATIONS:
                    await conn.execute(text(stmt))
            tables = list(Base.metadata.tables)
            log.info("db_initialized", tables=len(tables))
            return
        except Exception as exc:
            if attempt == retries:
                log.error("db_init_failed", attempt=attempt, error=str(exc))
                raise
            log.warning("db_not_ready", attempt=attempt, retry_in=delay)
            await asyncio.sleep(delay)
