"""Unit tests for the combined Google MCP server — upstream HTTP fully faked, no real API.

The fake client routes by (method, url-fragment) so one client can serve the multi-call flows
(search → fetch message; create_event → clash check → create). resolve_user_token is faked and
asserts the single "google" provider key is used for both Gmail and Calendar.
"""

from __future__ import annotations

import contextlib

import pytest

from app.mcp.servers import google_server as gs


class _FakeResponse:
    def __init__(self, payload: object, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _FakeClient:
    """Routes GET/POST/PATCH by a (method, url-fragment) -> payload table."""

    def __init__(self, routes: dict[tuple[str, str], object]) -> None:
        self._routes = routes
        self.calls: list[tuple[str, str]] = []

    def _respond(self, method: str, url: str) -> _FakeResponse:
        self.calls.append((method, url))
        for (m, frag), payload in self._routes.items():
            if m == method and frag in url:
                return _FakeResponse(payload)
        return _FakeResponse({})

    async def get(self, url, params=None, headers=None):
        return self._respond("GET", url)

    async def post(self, url, json=None, headers=None):
        return self._respond("POST", url)

    async def patch(self, url, json=None, headers=None):
        return self._respond("PATCH", url)


@pytest.fixture
def env(monkeypatch):
    seen: dict[str, str] = {}

    async def fake_resolve(user_id: str, provider: str) -> str:
        seen["provider"] = provider
        return "tok"

    monkeypatch.setattr(gs, "resolve_user_token", fake_resolve)

    def install(routes: dict[tuple[str, str], object]) -> _FakeClient:
        client = _FakeClient(routes)

        @contextlib.asynccontextmanager
        async def fake_http_client(timeout: int = 15):
            yield client

        monkeypatch.setattr(gs, "http_client", fake_http_client)
        return client

    return install, seen


async def test_send_email_uses_google_provider(env):
    install, seen = env
    install({("POST", "/messages/send"): {"id": "m1"}})
    result = await gs.send_email("u", "a@example.com", "hi", "body")
    assert result == {"id": "m1"}
    assert seen["provider"] == "google"  # one token key for Gmail + Calendar


async def test_search_email_fetches_each_message(env):
    install, _ = env
    # Detail route is listed first so it wins for ".../messages/x1"; the bare list URL
    # (".../messages", no query string in the url itself) then falls through to "/messages".
    install({("GET", "/messages/x1"): {"threadId": "t1", "snippet": "hello",
                                       "payload": {"headers": [{"name": "From", "value": "a@b.c"}]}},
             ("GET", "/messages"): {"messages": [{"id": "x1"}]}})
    result = await gs.search_email("u", "is:unread")
    assert len(result["messages"]) == 1
    assert result["messages"][0]["from"] == "a@b.c"


async def test_create_event_no_clash_creates(env):
    install, _ = env
    install({("GET", "/events"): {"items": []},          # no overlap
             ("POST", "/events"): {"id": "ev1"}})
    result = await gs.create_event("u", "Sync", "2030-01-01T10:00:00+05:30",
                                   "2030-01-01T10:30:00+05:30")
    assert result == {"id": "ev1"}


async def test_create_event_clash_returns_conflict(env):
    install, _ = env
    overlap = {"items": [{"id": "busy1", "summary": "Standup",
                          "start": {"dateTime": "2030-01-01T10:00:00+05:30"},
                          "end": {"dateTime": "2030-01-01T10:30:00+05:30"}}]}
    install({("GET", "/events"): overlap,
             ("POST", "/freeBusy"): {"calendars": {"primary": {"busy": []}}}})
    result = await gs.create_event("u", "Sync", "2030-01-01T10:00:00+05:30",
                                   "2030-01-01T10:30:00+05:30")
    assert result["conflict"] is True
    assert result["summary"] == "Sync"
