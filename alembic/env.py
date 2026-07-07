"""Alembic environment — async engine, URL + metadata sourced from the app.

Run migrations with ``alembic upgrade head``. The URL comes from app settings
(DATABASE_URL / .env) and the target metadata is the ORM ``Base.metadata``, so
``alembic revision --autogenerate`` stays in sync with app/data/models.py.
"""

from __future__ import annotations

import asyncio
import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.data.models import Base

config = context.config
# Configure logging from alembic.ini ONLY when alembic runs standalone (CLI).
# When the API invokes migrations in-process at startup (init_db), the root
# logger already carries structlog's JSON handler — fileConfig would REPLACE it
# and raise the root level to WARNING (alembic.ini [logger_root]), silently
# killing every app INFO log for the life of the process.
if config.config_file_name is not None and not logging.getLogger().handlers:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    """Database URL from app settings (asyncpg driver)."""
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection (``--sql`` mode)."""
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Open an async engine and run migrations against a live database."""
    engine = create_async_engine(_url(), pool_pre_ping=True)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
