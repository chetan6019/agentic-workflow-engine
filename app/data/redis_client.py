"""Redis async client singleton and the request rate limiter."""

from __future__ import annotations

import time
from functools import lru_cache

import structlog

import redis.asyncio as aioredis

from app.config import get_settings

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_redis() -> aioredis.Redis:
    """Return the cached Redis async client."""
    url = get_settings().redis_url
    return aioredis.from_url(url, decode_responses=True)


async def check_rate_limit(identity: str, limit: int, window_sec: int = 60) -> bool:
    """Fixed-window limiter. Return True if `identity` is within `limit` this window.

    INCR a per-window counter and set its TTL on first hit; the key expires with
    the window so no cleanup is needed. Fails open: a Redis error returns True
    rather than locking every caller out.
    """
    try:
        redis = get_redis()
        key = f"ratelimit:{identity}:{int(time.time()) // window_sec}"
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, window_sec)
        return count <= limit
    except Exception as exc:
        log.warning("rate_limit_check_failed", identity=identity, error=str(exc))
        return True
