"""Unit tests for the Finnhub MCP server — upstream HTTP fully faked, no real API.

The fake http_client yields a fake client whose ``.get`` returns canned JSON, so the real
``_shared.get_json`` and the server's query-building run unchanged. Redis + settings are faked.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from app.mcp.servers import finnhub_server as fh


class _FakeResponse:
    """Stand-in for an httpx.Response: just the two methods get_json touches."""

    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _FakeClient:
    """Records every GET and returns the same canned payload."""

    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.calls: list[dict] = []

    async def get(self, url, params=None, headers=None):
        self.calls.append({"url": url, "params": params})
        return _FakeResponse(self._payload)


class _FakeRedis:
    """Dict-backed INCRBY/EXPIRE stand-in for the per-minute rate counter."""

    def __init__(self) -> None:
        self.store: dict[str, int] = {}

    async def incrby(self, key: str, amount: int) -> int:
        self.store[key] = self.store.get(key, 0) + amount
        return self.store[key]

    async def expire(self, key: str, ttl: int) -> None:
        return None


@pytest.fixture
def fake_env(monkeypatch):
    """Patch settings (API key) and Redis; return a setter that installs a fake http client."""
    monkeypatch.setattr(fh, "get_settings",
                        lambda: SimpleNamespace(finnhub_api_key="testkey"))
    monkeypatch.setattr(fh, "get_redis", lambda: _FakeRedis())

    def install_client(payload: object) -> _FakeClient:
        client = _FakeClient(payload)

        @contextlib.asynccontextmanager
        async def fake_http_client(timeout: int = 15):
            yield client

        monkeypatch.setattr(fh, "http_client", fake_http_client)
        return client

    return install_client


async def test_get_quote_hits_quote_endpoint_with_token(fake_env):
    client = fake_env({"c": 150.0, "o": 148.0})
    result = await fh.get_quote("aapl")
    assert result["symbol"] == "AAPL"
    assert result["quote"] == {"c": 150.0, "o": 148.0}
    # URL + auth token + uppercased symbol were built correctly.
    assert client.calls[0]["url"].endswith("/quote")
    assert client.calls[0]["params"] == {"symbol": "AAPL", "token": "testkey"}


async def test_search_symbol_returns_results(fake_env):
    fake_env({"count": 1, "result": [{"symbol": "AAPL"}]})
    result = await fh.search_symbol("apple")
    assert result["query"] == "apple"
    assert result["results"]["result"][0]["symbol"] == "AAPL"


async def test_company_news_wraps_list(fake_env):
    fake_env([{"headline": "h1"}, {"headline": "h2"}])
    result = await fh.get_company_news("AAPL", "2026-01-01", "2026-01-31")
    assert len(result["news"]) == 2


async def test_rate_limit_raises(fake_env, monkeypatch):
    fake_env({"c": 1.0})
    monkeypatch.setattr(fh, "_RATE_PER_MINUTE", 0)  # every call is immediately over the cap
    with pytest.raises(RuntimeError, match="finnhub_rate_limited"):
        await fh.get_quote("AAPL")


async def test_missing_api_key_raises(fake_env, monkeypatch):
    fake_env({"c": 1.0})
    monkeypatch.setattr(fh, "get_settings",
                        lambda: SimpleNamespace(finnhub_api_key=None))
    with pytest.raises(RuntimeError, match="finnhub_api_key_not_configured"):
        await fh.get_quote("AAPL")
