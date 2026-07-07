"""Unit tests for the Reddit MCP server — upstream HTTP fully faked, no real API.

Reads patch ``_app_token`` directly (the token-fetch path is not under test here) and fake the
GET; the credential-missing paths exercise the real token helpers with faked settings.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from app.mcp.servers import reddit_server as rd


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _FakeClient:
    """Records GET/POST calls and returns the same canned payload."""

    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.gets: list[dict] = []
        self.posts: list[dict] = []

    async def get(self, url, params=None, headers=None):
        self.gets.append({"url": url, "params": params, "headers": headers})
        return _FakeResponse(self._payload)

    async def post(self, url, data=None, headers=None):
        self.posts.append({"url": url, "data": data, "headers": headers})
        return _FakeResponse(self._payload)


def _install_client(monkeypatch, payload: object) -> _FakeClient:
    client = _FakeClient(payload)

    @contextlib.asynccontextmanager
    async def fake_http_client(timeout: int = 15):
        yield client

    monkeypatch.setattr(rd, "http_client", fake_http_client)
    return client


async def test_search_subreddit_reads_with_bearer(monkeypatch):
    async def fake_app_token() -> str:
        return "apptoken"

    monkeypatch.setattr(rd, "_app_token", fake_app_token)
    client = _install_client(monkeypatch, {"data": {"children": [{"x": 1}]}})

    result = await rd.search_subreddit("python", "asyncio", limit=5)

    assert result["subreddit"] == "python"
    assert client.gets[0]["url"].endswith("/r/python/search")
    assert client.gets[0]["params"] == {"q": "asyncio", "restrict_sr": 1, "limit": 5}
    assert client.gets[0]["headers"]["Authorization"] == "Bearer apptoken"


async def test_get_post_builds_by_id_path(monkeypatch):
    async def fake_app_token() -> str:
        return "apptoken"

    monkeypatch.setattr(rd, "_app_token", fake_app_token)
    client = _install_client(monkeypatch, {"data": {}})

    await rd.get_post("abc123")
    assert client.gets[0]["url"].endswith("/by_id/t3_abc123")


async def test_post_comment_uses_user_token(monkeypatch):
    async def fake_user_token() -> str:
        return "usertoken"

    monkeypatch.setattr(rd, "_user_token", fake_user_token)
    client = _install_client(monkeypatch, {"json": {"errors": []}})

    await rd.post_comment("t3_abc", "nice post")
    assert client.posts[0]["url"].endswith("/api/comment")
    assert client.posts[0]["data"]["thing_id"] == "t3_abc"
    assert client.posts[0]["headers"]["Authorization"] == "Bearer usertoken"


async def test_missing_client_creds_raises(monkeypatch):
    monkeypatch.setattr(rd, "get_settings",
                        lambda: SimpleNamespace(reddit_client_id=None, reddit_client_secret=None))
    with pytest.raises(RuntimeError, match="reddit_client_credentials_not_configured"):
        await rd.search_subreddit("python", "asyncio")


async def test_missing_user_creds_raises(monkeypatch):
    monkeypatch.setattr(rd, "get_settings",
                        lambda: SimpleNamespace(reddit_client_id="id", reddit_client_secret="sec",
                                                reddit_username=None, reddit_password=None))
    with pytest.raises(RuntimeError, match="reddit_user_credentials_not_configured"):
        await rd.post_comment("t3_abc", "hi")
