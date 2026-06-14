"""Rate-limiter tests (REVIEW.md R14) over a dict-backed fake Redis."""

from __future__ import annotations

import pytest

from app.data import redis_client


class _FakeRedis:
    """Minimal INCR/EXPIRE stand-in; optionally raises to test fail-open."""

    def __init__(self, *, broken: bool = False):
        self.store: dict[str, int] = {}
        self.expiries: dict[str, int] = {}
        self._broken = broken

    async def incr(self, key):
        if self._broken:
            raise RuntimeError("redis down")
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    async def expire(self, key, ttl):
        self.expiries[key] = ttl


@pytest.fixture
def fake_redis(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(redis_client, "get_redis", lambda: r)
    return r


async def test_allows_up_to_limit_then_blocks(fake_redis):
    results = [await redis_client.check_rate_limit("u1", limit=3) for _ in range(4)]
    assert results == [True, True, True, False]


async def test_sets_ttl_on_first_hit_only(fake_redis):
    await redis_client.check_rate_limit("u1", limit=5)
    await redis_client.check_rate_limit("u1", limit=5)
    assert len(fake_redis.expiries) == 1  # expire set once per window


async def test_identities_are_isolated(fake_redis):
    assert await redis_client.check_rate_limit("a", limit=1) is True
    assert await redis_client.check_rate_limit("b", limit=1) is True
    assert await redis_client.check_rate_limit("a", limit=1) is False


async def test_fails_open_on_redis_error(monkeypatch):
    monkeypatch.setattr(redis_client, "get_redis", lambda: _FakeRedis(broken=True))
    assert await redis_client.check_rate_limit("u1", limit=1) is True
