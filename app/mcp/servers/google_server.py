"""MCP Google server — Gmail + Calendar on ONE Google OAuth token, on port 7008.

Supersedes the separate gmail_server (7002) and calendar_server (7001): both sets of tools now
live here and resolve a single token under the "google" provider key. delete_event is implemented
but HITL-gated BEFORE execution (orchestrator _APPROVAL_ACTIONS + policy.yaml approval_required_
actions), so an approved delete never runs unreviewed; delete_email stays intentionally NOT
implemented. The helper bodies mirror the original gmail/calendar servers (kept beginner-friendly,
internal helpers rather than new files, per the repo's file-count rule).
"""

from __future__ import annotations

import base64
import datetime
from zoneinfo import ZoneInfo

import structlog
from fastmcp import FastMCP

from app.config import get_settings
from app.mcp.servers._shared import http_client, resolve_user_token

log = structlog.get_logger(__name__)
mcp = FastMCP("google")
_PROVIDER = "google"  # single OAuth token granting both Gmail and Calendar scopes
_GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me"
_CAL = "https://www.googleapis.com/calendar/v3"
_TZ = get_settings().default_tz
_BODY_LIMIT = 2000
_WORK_START = datetime.time(9, 0)
_WORK_END = datetime.time(18, 0)
_SLOT_SEARCH_DAYS = 7


# ── Gmail helpers ────────────────────────────────────────────────────────────
def _make_raw(to: str, subject: str, body: str) -> str:
    """Build a base64url-encoded RFC 2822 message."""
    msg = f"To: {to}\r\nSubject: {subject}\r\n\r\n{body}"
    return base64.urlsafe_b64encode(msg.encode()).decode().rstrip("=")


def _b64url(data: str) -> str:
    """Decode a Gmail base64url body part to text, best effort."""
    try:
        return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", "ignore")
    except Exception:
        return ""


def _headers_map(payload: dict) -> dict:
    """Map lowercased Gmail header names to their values."""
    return {h.get("name", "").lower(): h.get("value", "")
            for h in payload.get("headers", [])}


def _extract_body(payload: dict) -> str:
    """Best-effort plain-text body from a Gmail message payload."""
    data = (payload.get("body") or {}).get("data")
    if data:
        return _b64url(data)
    for part in payload.get("parts", []):
        if part.get("mimeType") == "text/plain":
            d = (part.get("body") or {}).get("data")
            if d:
                return _b64url(d)
    for part in payload.get("parts", []):
        nested = _extract_body(part)
        if nested:
            return nested
    return ""


@mcp.tool()
async def search_email(user_id: str, query: str, max_results: int = 10) -> dict:
    """Search Gmail and return message details. Build `query` with Gmail search operators.

    user_id is auto-injected. Default to `in:inbox category:primary is:unread` when no scope
    is given; use `from:`, `subject:`, `is:unread`, `newer_than:7d`, etc.
    """
    token = await resolve_user_token(user_id, _PROVIDER)
    hdr = {"Authorization": f"Bearer {token}"}
    async with http_client() as c:
        lst = await c.get(f"{_GMAIL}/messages",
                          params={"q": query, "maxResults": max_results}, headers=hdr)
        lst.raise_for_status()
        messages = []
        for ref in lst.json().get("messages", []):
            mid = ref.get("id")
            if not mid:
                continue
            r = await c.get(f"{_GMAIL}/messages/{mid}", params={"format": "full"}, headers=hdr)
            if r.status_code != 200:
                continue
            full = r.json()
            h = _headers_map(full.get("payload", {}))
            messages.append({"id": mid, "thread_id": full.get("threadId", ""),
                             "from": h.get("from", ""), "subject": h.get("subject", ""),
                             "date": h.get("date", ""), "snippet": full.get("snippet", ""),
                             "body": _extract_body(full.get("payload", {}))[:_BODY_LIMIT]})
        log.info("google_gmail_searched", user_id=user_id, count=len(messages))
        return {"query": query, "messages": messages}


@mcp.tool()
async def send_email(user_id: str, to: str, subject: str, body: str) -> dict:
    """Send an email via Gmail. `to` MUST be a valid email address. Args auto-inject user_id."""
    token = await resolve_user_token(user_id, _PROVIDER)
    async with http_client() as c:
        r = await c.post(f"{_GMAIL}/messages/send", json={"raw": _make_raw(to, subject, body)},
                         headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.info("google_email_sent", user_id=user_id, to=to)
        return r.json()


@mcp.tool()
async def create_draft(user_id: str, to: str, subject: str, body: str) -> dict:
    """Create a Gmail draft. Args: user_id (auto-injected), to, subject, body."""
    token = await resolve_user_token(user_id, _PROVIDER)
    async with http_client() as c:
        r = await c.post(f"{_GMAIL}/drafts",
                         json={"message": {"raw": _make_raw(to, subject, body)}},
                         headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.info("google_draft_created", user_id=user_id)
        return r.json()


@mcp.tool()
async def get_thread(user_id: str, thread_id: str) -> dict:
    """Fetch a Gmail thread's messages (from/subject/snippet). Args: user_id, thread_id."""
    token = await resolve_user_token(user_id, _PROVIDER)
    async with http_client() as c:
        r = await c.get(f"{_GMAIL}/threads/{thread_id}", params={"format": "full"},
                        headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        out = []
        for m in r.json().get("messages", []):
            h = _headers_map(m.get("payload", {}))
            out.append({"id": m.get("id", ""), "from": h.get("from", ""),
                        "subject": h.get("subject", ""), "snippet": m.get("snippet", "")})
        log.info("google_thread_fetched", user_id=user_id, thread_id=thread_id, count=len(out))
        return {"thread_id": thread_id, "messages": out}


# ── Calendar helpers ─────────────────────────────────────────────────────────
async def _overlapping_events(c, token: str, calendar_id: str, start: str, end: str) -> list[dict]:
    """Return timed events overlapping [start, end]; Google filters by the window."""
    params = {"timeMin": start, "timeMax": end, "singleEvents": "true",
              "orderBy": "startTime", "timeZone": _TZ}
    r = await c.get(f"{_CAL}/calendars/{calendar_id}/events", params=params,
                    headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    out = []
    for ev in r.json().get("items", []):
        s, e = ev.get("start", {}).get("dateTime"), ev.get("end", {}).get("dateTime")
        if s and e:  # skip all-day events (date-only, no dateTime)
            out.append({"id": ev.get("id"), "summary": ev.get("summary", "(busy)"),
                        "start": s, "end": e})
    return out


async def _next_free_slot(c, token: str, calendar_id: str, start: str, end: str) -> dict | None:
    """Find the earliest free working-hours slot of the same duration within 7 days."""
    tz = ZoneInfo(_TZ)
    start_dt = datetime.datetime.fromisoformat(start)
    duration = datetime.datetime.fromisoformat(end) - start_dt
    search_from = max(start_dt.astimezone(tz), datetime.datetime.now(tz))
    search_to = search_from + datetime.timedelta(days=_SLOT_SEARCH_DAYS)
    body = {"timeMin": search_from.isoformat(), "timeMax": search_to.isoformat(),
            "timeZone": _TZ, "items": [{"id": calendar_id}]}
    r = await c.post(f"{_CAL}/freeBusy", json=body, headers={"Authorization": f"Bearer {token}"})
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


async def _clash_or_none(c, token: str, calendar_id: str, start: str, end: str,
                         *, exclude_id: str | None = None) -> dict | None:
    """Return a conflict payload (with the next free slot) if [start, end] overlaps, else None."""
    clashing = [e for e in await _overlapping_events(c, token, calendar_id, start, end)
                if e.get("id") != exclude_id]
    if not clashing:
        return None
    suggested = await _next_free_slot(c, token, calendar_id, start, end)
    return {"conflict": True, "requested": {"start": start, "end": end},
            "conflicting": clashing, "suggested": suggested}


@mcp.tool()
async def create_event(user_id: str, summary: str, start: str, end: str,
                       calendar_id: str = "primary", description: str = "") -> dict:
    """Create a calendar event (ISO 8601 start/end). On a clash, returns a conflict payload
    (with the next free slot) for HITL instead of creating the event. Args auto-inject user_id.
    """
    token = await resolve_user_token(user_id, _PROVIDER)
    async with http_client() as c:
        conflict = await _clash_or_none(c, token, calendar_id, start, end)
        if conflict:
            log.info("google_event_conflict", user_id=user_id,
                     has_suggestion=conflict["suggested"] is not None)
            return {**conflict, "summary": summary}
        body = {"summary": summary, "start": {"dateTime": start, "timeZone": _TZ},
                "end": {"dateTime": end, "timeZone": _TZ}, "description": description}
        r = await c.post(f"{_CAL}/calendars/{calendar_id}/events", json=body,
                         headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.info("google_event_created", user_id=user_id)
        return r.json()


async def _resolve_event(c, token: str, calendar_id: str, match_summary: str,
                         time_min: str, time_max: str) -> dict:
    """Find the one event whose summary matches within [time_min, time_max] and return it.

    Lets callers update an event they can only describe (e.g. "the 1:1 with Rakesh tomorrow")
    when no event_id is known — the planner is told to pass match_summary + a day window instead
    of inventing an id (see prompts.py). Returns the full event (id + current times) so the update
    can clash-check its new window. Raises on a missing window or a no/ambiguous match, so we
    surface the failure instead of patching the wrong event.
    """
    if not (match_summary.strip() and time_min and time_max):
        raise ValueError("event_id_or_match_summary_window_required")
    params = {"timeMin": time_min, "timeMax": time_max, "singleEvents": "true",
              "orderBy": "startTime", "timeZone": _TZ}
    r = await c.get(f"{_CAL}/calendars/{calendar_id}/events", params=params,
                    headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    needle = match_summary.strip().lower()
    matches = [ev for ev in r.json().get("items", []) if needle in ev.get("summary", "").lower()]
    if len(matches) != 1:
        raise ValueError(f"no_unique_event_match:{match_summary!r} ({len(matches)} found)")
    return matches[0]


async def _get_event(c, token: str, calendar_id: str, event_id: str) -> dict:
    """Fetch a single event by id, so a direct-id update can read its current times."""
    r = await c.get(f"{_CAL}/calendars/{calendar_id}/events/{event_id}",
                    headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return r.json()


@mcp.tool()
async def update_event(user_id: str, event_id: str | None = None, calendar_id: str = "primary",
                       summary: str | None = None, start: str | None = None,
                       end: str | None = None, match_summary: str | None = None,
                       time_min: str | None = None, time_max: str | None = None) -> dict:
    """Patch an existing calendar event.

    Pass ``event_id`` directly, OR locate the event by ``match_summary`` (distinctive words from
    the title, e.g. "Rakesh") plus a ``time_min``/``time_max`` window (ISO 8601) bounding the day
    it falls on, when no id is known. ``summary``/``start``/``end`` are the new values to write
    (all optional). On a clash with the new window, returns a conflict payload for HITL instead of
    patching. Args auto-inject user_id.
    """
    token = await resolve_user_token(user_id, _PROVIDER)
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
        # start/end may be partial (e.g. extending only the end), so fall back to the event's
        # current times for the missing side when clash-checking the effective new window.
        new_start = start or event.get("start", {}).get("dateTime")
        new_end = end or event.get("end", {}).get("dateTime")
        if new_start and new_end:
            conflict = await _clash_or_none(c, token, calendar_id, new_start, new_end,
                                            exclude_id=event_id)
            if conflict:
                log.info("google_update_conflict", user_id=user_id)
                return conflict
        r = await c.patch(f"{_CAL}/calendars/{calendar_id}/events/{event_id}", json=patch,
                          headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.info("google_event_updated", user_id=user_id)
        return r.json()

@mcp.tool()
async def delete_event(user_id: str, event_id: str | None = None, calendar_id: str = "primary",
                       match_summary: str | None = None, time_min: str | None = None,
                       time_max: str | None = None) -> dict:
    """Delete a calendar event.

    Pass ``event_id`` directly, or locate it by ``match_summary`` plus a ``time_min``/``time_max``
    window (ISO 8601) when no id is known. Returns the deleted event's summary + start/end so the
    confirmation can name the event's ACTUAL date: the planner's time_min is only a SEARCH window
    (often a whole day or week), so echoing it would show the wrong date.
    """
    token = await resolve_user_token(user_id, _PROVIDER)
    async with http_client() as c:
        # Fetch the event in BOTH paths so the return carries its real times, not the search window.
        if event_id is None:
            event = await _resolve_event(c, token, calendar_id,
                                         match_summary or "", time_min or "", time_max or "")
        else:
            event = await _get_event(c, token, calendar_id, event_id)
        event_id = event["id"]
        r = await c.delete(f"{_CAL}/calendars/{calendar_id}/events/{event_id}",
                           headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.info("google_event_deleted", user_id=user_id)
        return {"deleted": event_id, "summary": event.get("summary", ""),
                "start": event.get("start", {}).get("dateTime"),
                "end": event.get("end", {}).get("dateTime")}


@mcp.tool()
async def find_free_slot(user_id: str, time_min: str, time_max: str,
                         duration_minutes: int = 30) -> dict:
    """Query freebusy for available slots. Args: user_id (auto-injected), time_min, time_max."""
    token = await resolve_user_token(user_id, _PROVIDER)
    body = {"timeMin": time_min, "timeMax": time_max, "timeZone": _TZ,
            "items": [{"id": "primary"}]}
    async with http_client() as c:
        r = await c.post(f"{_CAL}/freeBusy", json=body,
                         headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.debug("google_freebusy_queried", user_id=user_id)
        return r.json()


@mcp.tool()
async def list_upcoming(user_id: str, calendar_id: str = "primary", max_results: int = 10,
                        time_min: str | None = None, time_max: str | None = None) -> dict:
    """List calendar events, soonest first.

    Defaults to events from now onward; pass ``time_min``/``time_max`` (ISO 8601) to scope to a
    date range — e.g. "what's on my calendar tomorrow" or "events next week". Args auto-inject
    user_id. (Without these params a date-scoped request couldn't be honoured and the call failed
    when the planner supplied them.)
    """
    token = await resolve_user_token(user_id, _PROVIDER)
    # Lower bound defaults to "now" so a bare call lists upcoming events; an explicit time_min
    # (e.g. the start of a requested day) scopes the list to that range instead.
    start = time_min or datetime.datetime.now(datetime.timezone.utc).isoformat()
    params = {"maxResults": max_results, "singleEvents": "true", "orderBy": "startTime",
              "timeMin": start, "timeZone": _TZ}
    if time_max:
        params["timeMax"] = time_max
    async with http_client() as c:
        r = await c.get(f"{_CAL}/calendars/{calendar_id}/events", params=params,
                        headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        return r.json()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=7008, path="/mcp")
