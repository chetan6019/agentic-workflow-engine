"""Redis async client singleton and idempotency helpers."""

from __future__ import annotations

import structlog
from functools import lru_cache

import redis.asyncio as aioredis

from app.config import get_settings

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_redis() -> aioredis.Redis:
    """Return the cached Redis async client."""
    url = get_settings().redis_url
    return aioredis.from_url(url, decode_responses=True)


async def set_idempotency_key(key: str, value: str = "1", ttl: int = 600) -> None:
    """Set an idempotency key with a TTL in seconds."""
    await get_redis().set(f"idem:{key}", value, ex=ttl)
    log.debug("idempotency_key_set", key=key, ttl=ttl)


async def check_idempotency_key(key: str) -> bool:
    """Return True if the idempotency key exists in Redis."""
    exists = bool(await get_redis().exists(f"idem:{key}"))
    log.debug("idempotency_key_checked", key=key, exists=exists)
    return exists
