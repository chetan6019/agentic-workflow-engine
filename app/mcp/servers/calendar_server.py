# STALE (2026-06-22): superseded by google_server.py (Gmail + Calendar on one token, port 7008);
# excluded from docker-compose and the MCP client registry. Kept in place per CLAUDE.md R13.
"""MCP Calendar server — deterministic Google Calendar adapter on port 7001."""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import structlog
from fastmcp import FastMCP

from app.config import get_settings
from app.mcp.servers._shared import http_client, resolve_user_token

log = structlog.get_logger(__name__)
mcp = FastMCP("calendar")
_BASE = "https://www.googleapis.com/calendar/v3"
# Zone Google should interpret/return event times in, so responses aren't UTC.
_TZ = get_settings().default_tz
# Working-hours window used when proposing the next free slot for a clashing event.
_WORK_START = datetime.time(9, 0)
_WORK_END = datetime.time(18, 0)
_SLOT_SEARCH_DAYS = 7


async def _overlapping_events(c, token: str, calendar_id: str, start: str, end: str) -> list[dict]:
    """Return timed events overlapping [start, end]; Google already filters by the window."""
    params = {"timeMin": start, "timeMax": end, "singleEvents": "true",
              "orderBy": "startTime", "timeZone": _TZ}
    r = await c.get(f"{_BASE}/calendars/{calendar_id}/events", params=params,
                    headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    out = []
    for ev in r.json().get("items", []):
        s, e = ev.get("start", {}).get("dateTime"), ev.get("end", {}).get("dateTime")
        if s and e:  # skip all-day events (date-only, no dateTime)
            out.append({"id": ev.get("id"), "summary": ev.get("summary", "(busy)"),
                        "start": s, "end": e})
    return out


async def _clash_or_none(c, token: str, calendar_id: str, start: str, end: str,
                         *, exclude_id: str | None = None) -> dict | None:
    """Return a conflict payload if [start, end] overlaps another event, else None.

    ``exclude_id`` omits one event (the one being updated) so moving or extending it
    does not clash with itself. The payload shape matches what the orchestrator's
    conflict path expects, including the next free working-hours slot.
    """
    clashing = [e for e in await _overlapping_events(c, token, calendar_id, start, end)
                if e.get("id") != exclude_id]
    if not clashing:
        return None
    suggested = await _next_free_slot(c, token, calendar_id, start, end)
    return {"conflict": True, "requested": {"start": start, "end": end},
            "conflicting": clashing, "suggested": suggested}


async def _next_free_slot(c, token: str, calendar_id: str, start: str, end: str) -> dict | None:
    """Find the earliest free working-hours slot of the same duration within 7 days."""
    tz = ZoneInfo(_TZ)
    start_dt = datetime.datetime.fromisoformat(start)
    duration = datetime.datetime.fromisoformat(end) - start_dt
    search_from = max(start_dt.astimezone(tz), datetime.datetime.now(tz))
    search_to = search_from + datetime.timedelta(days=_SLOT_SEARCH_DAYS)
    body = {"timeMin": search_from.isoformat(), "timeMax": search_to.isoformat(),
            "timeZone": _TZ, "items": [{"id": calendar_id}]}
    r = await c.post(f"{_BASE}/freeBusy", json=body, headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    busy = [(datetime.datetime.fromisoformat(b["start"]), datetime.datetime.fromisoformat(b["end"]))
            for b in r.json().get("calendars", {}).get(calendar_id, {}).get("busy", [])]
    for day_offset in range(_SLOT_SEARCH_DAYS):
        d = (search_from + datetime.timedelta(days=day_offset)).date()
        if d.weekday() >= 5:  # skip Sat/Sun
            continue
        win_start = datetime.datetime.combine(d, _WORK_START, tzinfo=tz)
        win_end = datetime.datetime.combine(d, _WORK_END, tzinfo=tz)
        cursor = max(win_start, search_from)
        day_busy = sorted((max(b0, win_start), min(b1, win_end))
                          for b0, b1 in busy if b1 > win_start and b0 < win_end)
        for b0, b1 in day_busy:
            if b0 - cursor >= duration:
                return {"start": cursor.isoformat(), "end": (cursor + duration).isoformat()}
            cursor = max(cursor, b1)
        if win_end - cursor >= duration:
            return {"start": cursor.isoformat(), "end": (cursor + duration).isoformat()}
    return None


@mcp.tool()
async def create_event(user_id: str, summary: str, start: str, end: str,
                       calendar_id: str = "primary", description: str = "") -> dict:
    """Create a calendar event. Args: user_id (auto-injected), summary, start, end (ISO 8601).

    If the requested slot clashes with an existing event, no event is created; instead a
    ``conflict`` payload with the next free working-hours slot is returned for HITL approval.
    """
    token = await resolve_user_token(user_id, "calendar")
    async with http_client() as c:
        conflict = await _clash_or_none(c, token, calendar_id, start, end)
        if conflict:
            log.info("calendar_event_conflict", user_id=user_id,
                     has_suggestion=conflict["suggested"] is not None)
            return {**conflict, "summary": summary}
        body = {"summary": summary, "start": {"dateTime": start, "timeZone": _TZ},
                "end": {"dateTime": end, "timeZone": _TZ}, "description": description}
        r = await c.post(f"{_BASE}/calendars/{calendar_id}/events", json=body,
                         headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.info("calendar_event_created", user_id=user_id)
        return r.json()


async def _resolve_event(c, token: str, calendar_id: str, match_summary: str,
                         time_min: str, time_max: str) -> dict:
    """Find the one event whose summary matches within [time_min, time_max] and return it.

    Lets callers update/delete an event they can only describe (e.g. "the 1:1 with
    Priya tomorrow") when no event_id is known. Returns the full event (id + current
    times) so an update can clash-check its new window. Raises on a missing window or a
    no/ambiguous match, so we surface the failure instead of touching the wrong event.
    """
    if not (match_summary.strip() and time_min and time_max):
        raise ValueError("event_id_or_match_summary_window_required")
    params = {"timeMin": time_min, "timeMax": time_max, "singleEvents": "true",
              "orderBy": "startTime", "timeZone": _TZ}
    r = await c.get(f"{_BASE}/calendars/{calendar_id}/events", params=params,
                    headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    needle = match_summary.strip().lower()
    matches = [ev for ev in r.json().get("items", []) if needle in ev.get("summary", "").lower()]
    if len(matches) != 1:
        raise ValueError(f"no_unique_event_match:{match_summary!r} ({len(matches)} found)")
    return matches[0]


async def _get_event(c, token: str, calendar_id: str, event_id: str) -> dict:
    """Fetch a single event by id, so a direct-id update can read its current times."""
    r = await c.get(f"{_BASE}/calendars/{calendar_id}/events/{event_id}",
                    headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return r.json()


@mcp.tool()
async def update_event(user_id: str, event_id: str | None = None, calendar_id: str = "primary",
                       summary: str | None = None, start: str | None = None,
                       end: str | None = None, match_summary: str | None = None,
                       time_min: str | None = None, time_max: str | None = None) -> dict:
    """Patch an existing calendar event.

    Pass ``event_id`` directly, or locate the event by ``match_summary`` plus a
    ``time_min``/``time_max`` window (ISO 8601) when no id is known. ``summary``/
    ``start``/``end`` are the new values to write (all optional). If the new time clashes
    with another event, nothing is patched; a ``conflict`` payload is returned for HITL.
    """
    token = await resolve_user_token(user_id, "calendar")
    patch = {k: v for k, v in {"summary": summary,
                               "start": start and {"dateTime": start, "timeZone": _TZ},
                               "end": end and {"dateTime": end, "timeZone": _TZ}}.items() if v}
    async with http_client() as c:
        if event_id is None:
            event = await _resolve_event(c, token, calendar_id,
                                         match_summary or "", time_min or "", time_max or "")
        else:
            event = await _get_event(c, token, calendar_id, event_id)
        event_id = event["id"]
        # Clash-check the effective new window. start/end may be partial (e.g. an extend
        # sets only end), so fall back to the event's current times for the missing side.
        new_start = start or event.get("start", {}).get("dateTime")
        new_end = end or event.get("end", {}).get("dateTime")
        if new_start and new_end:
            conflict = await _clash_or_none(c, token, calendar_id, new_start, new_end,
                                            exclude_id=event_id)
            if conflict:
                log.info("calendar_update_conflict", user_id=user_id,
                         has_suggestion=conflict["suggested"] is not None)
                return conflict
        r = await c.patch(f"{_BASE}/calendars/{calendar_id}/events/{event_id}",
                          json=patch, headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.info("calendar_event_updated", user_id=user_id)
        return r.json()


@mcp.tool()
async def delete_event(user_id: str, event_id: str | None = None, calendar_id: str = "primary",
                       match_summary: str | None = None, time_min: str | None = None,
                       time_max: str | None = None) -> dict:
    """Delete a calendar event.

    Pass ``event_id`` directly, or locate it by ``match_summary`` plus a
    ``time_min``/``time_max`` window (ISO 8601) when no id is known.
    """
    token = await resolve_user_token(user_id, "calendar")
    async with http_client() as c:
        if event_id is None:
            event = await _resolve_event(c, token, calendar_id,
                                         match_summary or "", time_min or "", time_max or "")
            event_id = event["id"]
        r = await c.delete(f"{_BASE}/calendars/{calendar_id}/events/{event_id}",
                           headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.info("calendar_event_deleted", user_id=user_id)
        return {"deleted": event_id}


@mcp.tool()
async def find_free_slot(user_id: str, time_min: str, time_max: str,
                         duration_minutes: int = 30) -> dict:
    """Query freebusy for available slots. Args: user_id (auto-injected), time_min, time_max."""
    token = await resolve_user_token(user_id, "calendar")
    body = {"timeMin": time_min, "timeMax": time_max, "timeZone": _TZ,
            "items": [{"id": "primary"}]}
    async with http_client() as c:
        r = await c.post(f"{_BASE}/freeBusy", json=body,
                         headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.debug("calendar_freebusy_queried", user_id=user_id)
        return r.json()


@mcp.tool()
async def list_upcoming(user_id: str, calendar_id: str = "primary",
                        max_results: int = 10) -> dict:
    """List upcoming events. Args: user_id (auto-injected), calendar_id, max_results."""
    token = await resolve_user_token(user_id, "calendar")
    # Google Calendar requires timeMin as an RFC3339 timestamp; a tz-aware UTC
    # isoformat already ends with +00:00, so don't also append "Z".
    now_rfc3339 = datetime.datetime.now(datetime.timezone.utc).isoformat()
    params = {"maxResults": max_results, "singleEvents": "true",
              "orderBy": "startTime", "timeMin": now_rfc3339, "timeZone": _TZ}
    async with http_client() as c:
        r = await c.get(f"{_BASE}/calendars/{calendar_id}/events", params=params,
                        headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=7001, path="/mcp")
