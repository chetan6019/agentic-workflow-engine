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
