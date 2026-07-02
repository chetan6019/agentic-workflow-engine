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
        self.last_get_params: dict = {}

    def _respond(self, method: str, url: str) -> _FakeResponse:
        self.calls.append((method, url))
        for (m, frag), payload in self._routes.items():
            if m == method and frag in url:
                return _FakeResponse(payload)
        return _FakeResponse({})

    async def get(self, url, params=None, headers=None):
        self.last_get_params = params or {}
        return self._respond("GET", url)

    async def post(self, url, json=None, headers=None):
        return self._respond("POST", url)

    async def patch(self, url, json=None, headers=None):
        return self._respond("PATCH", url)

    async def delete(self, url, headers=None):
        return self._respond("DELETE", url)


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


async def test_update_event_by_id_patches(env):
    install, _ = env
    # Direct-id path: GET current event (no clash on its window), then PATCH.
    event = {"id": "ev1", "summary": "1:1 with Rakesh",
             "start": {"dateTime": "2030-01-01T10:00:00+05:30"},
             "end": {"dateTime": "2030-01-01T10:30:00+05:30"}}
    install({("GET", "/events/ev1"): event,
             ("GET", "/events"): {"items": []},  # clash check: nothing overlaps
             ("PATCH", "/events/ev1"): {**event, "end": {"dateTime": "2030-01-01T11:00:00+05:30"}}})
    result = await gs.update_event("u", event_id="ev1", end="2030-01-01T11:00:00+05:30")
    assert result["id"] == "ev1"


async def test_update_event_by_match_summary_resolves_id(env):
    install, _ = env
    # Describe-by-summary path: no event_id given. The window list resolves the one matching
    # event, then the extended window is clash-checked and the event is PATCHed. Regression for
    # "the system needs the specific event ID" — the planner omits event_id by design.
    match = {"items": [{"id": "ev9", "summary": "1:1 with Rakesh",
                        "start": {"dateTime": "2030-01-01T10:00:00+05:30"},
                        "end": {"dateTime": "2030-01-01T10:30:00+05:30"}}]}
    client = install({("GET", "/events"): match,  # serves BOTH the resolve list and the clash check
                      ("PATCH", "/events/ev9"): {"id": "ev9", "summary": "1:1 with Rakesh"}})
    result = await gs.update_event(
        "u", match_summary="Rakesh", time_min="2030-01-01T00:00:00+05:30",
        time_max="2030-01-01T23:59:59+05:30", end="2030-01-01T11:00:00+05:30")
    assert result["id"] == "ev9"
    # The id was resolved from the summary and the PATCH hit that resolved id.
    patched = [url for (method, url) in client.calls if method == "PATCH"]
    assert any(url.endswith("/events/ev9") for url in patched)


async def test_update_event_without_id_or_match_raises(env):
    install, _ = env
    install({})
    with pytest.raises(ValueError, match="event_id_or_match_summary_window_required"):
        await gs.update_event("u", end="2030-01-01T11:00:00+05:30")


async def test_delete_event_by_match_summary_returns_actual_date(env):
    install, seen = env
    # delete_event resolves the id from the summary window, DELETEs it, and must use the single
    # "google" provider token (regression: a copy-paste left it resolving "calendar"). It must also
    # return the event's ACTUAL start (regression: "delete shows the wrong date"). The window below
    # spans a whole week but the event is on the 3rd — the confirmation must show the 3rd, not the
    # window's start on the 1st (which is all the old {"deleted": id} return left the composer).
    match = {"items": [{"id": "ev7", "summary": "1:1 with Rakesh",
                        "start": {"dateTime": "2030-01-03T10:00:00+05:30"},
                        "end": {"dateTime": "2030-01-03T10:30:00+05:30"}}]}
    install({("GET", "/events"): match})
    result = await gs.delete_event(
        "u", match_summary="Rakesh", time_min="2030-01-01T00:00:00+05:30",
        time_max="2030-01-07T23:59:59+05:30")
    assert result["deleted"] == "ev7"
    assert result["start"] == "2030-01-03T10:00:00+05:30"  # the real event date, not time_min
    assert result["summary"] == "1:1 with Rakesh"
    assert seen["provider"] == "google"


async def test_delete_event_by_id_returns_actual_date(env):
    install, _ = env
    # Direct-id path now fetches the event first so the return still carries its real start/end.
    event = {"id": "ev2", "summary": "Sync",
             "start": {"dateTime": "2030-02-01T09:00:00+05:30"},
             "end": {"dateTime": "2030-02-01T09:30:00+05:30"}}
    install({("GET", "/events/ev2"): event})  # DELETE falls through to the empty default route
    result = await gs.delete_event("u", event_id="ev2")
    assert result["deleted"] == "ev2"
    assert result["start"] == "2030-02-01T09:00:00+05:30"


async def test_list_upcoming_bare_defaults_to_now_unbounded(env):
    install, _ = env
    client = install({("GET", "/events"): {"items": [{"id": "ev1", "summary": "Sync"}]}})
    result = await gs.list_upcoming("u")
    assert result["items"][0]["id"] == "ev1"
    # A bare call still lists from "now" with no upper bound.
    assert "timeMin" in client.last_get_params
    assert "timeMax" not in client.last_get_params


async def test_list_upcoming_scopes_to_time_range(env):
    install, _ = env
    # Regression ("list event is failing"): a date-scoped list must accept time_min/time_max and
    # forward them to the query instead of rejecting them as unexpected args.
    client = install({("GET", "/events"): {"items": []}})
    await gs.list_upcoming("u", time_min="2030-01-01T00:00:00+05:30",
                           time_max="2030-01-07T23:59:59+05:30")
    assert client.last_get_params["timeMin"] == "2030-01-01T00:00:00+05:30"
    assert client.last_get_params["timeMax"] == "2030-01-07T23:59:59+05:30"
