"""Async SQLAlchemy engine, session factory, and Alembic-driven schema bootstrap.

Schema ownership: Alembic is the single source of truth (see ``alembic/``).
``init_db`` runs ``alembic upgrade head`` on startup, retrying until Postgres is
ready. Evolve the schema by adding Alembic revisions
(``alembic revision --autogenerate``) — never with ad-hoc DDL here.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

# Repo root holds alembic.ini and the alembic/ package; resolve absolutely so the
# config works regardless of the process's current working directory.
_ROOT = Path(__file__).resolve().parents[2]

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


def _alembic_config():
    """Build an Alembic Config pointing at the repo's alembic.ini + alembic/ dir."""
    from alembic.config import Config

    cfg = Config(str(_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_ROOT / "alembic"))
    return cfg


async def _is_pre_alembic_db() -> bool:
    """True if the app schema already exists but Alembic has never tracked it.

    Lets us adopt Alembic on a database first created by the old create_all path:
    stamp it to head instead of trying to re-create existing tables.
    """
    import asyncpg

    dsn = get_settings().database_url.replace("+asyncpg", "")
    conn = await asyncpg.connect(dsn)
    try:
        has_version = await conn.fetchval("SELECT to_regclass('public.alembic_version')")
        has_users = await conn.fetchval("SELECT to_regclass('public.users')")
        return has_users is not None and has_version is None
    finally:
        await conn.close()


def _migrate() -> None:
    """Bring the database to head (sync; runs in a worker thread, see init_db).

    env.py drives an async engine via asyncio.run, so this must NOT run inside the
    app's event loop — init_db calls it through asyncio.to_thread.
    """
    from alembic import command

    cfg = _alembic_config()
    if asyncio.run(_is_pre_alembic_db()):
        log.info("db_stamping_pre_alembic_schema")
        command.stamp(cfg, "head")
    else:
        command.upgrade(cfg, "head")


async def init_db(*, retries: int = 10, delay: float = 2.0) -> None:
    """Run Alembic migrations to head, retrying until Postgres is ready."""
    for attempt in range(1, retries + 1):
        try:
            await asyncio.to_thread(_migrate)
            log.info("db_migrated")
            return
        except Exception as exc:
            if attempt == retries:
                log.error("db_init_failed", attempt=attempt, error=str(exc))
                raise
            log.warning("db_not_ready", attempt=attempt, retry_in=delay)
            await asyncio.sleep(delay)
