# STALE (2026-06-22): superseded by google_server.py (Gmail + Calendar on one token, port 7008);
# excluded from docker-compose and the MCP client registry. Kept in place per CLAUDE.md R13.
"""MCP Gmail server — deterministic Gmail adapter on port 7002."""

from __future__ import annotations

import base64

import structlog
from fastmcp import FastMCP

from app.mcp.servers._shared import http_client, resolve_user_token

log = structlog.get_logger(__name__)
mcp = FastMCP("gmail")
_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
_PEOPLE_SEARCH = "https://people.googleapis.com/v1/people:searchContacts"
_BODY_LIMIT = 2000


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
    """Search Gmail and return full message details (sender, subject, date, snippet, body).

    Build `query` with Gmail search operators so results match the user's intent:
      - DEFAULT when no inbox/category is specified: `in:inbox category:primary is:unread`
      - by sender: `from:linkedin.com` (use this for "LinkedIn emails"), `from:zomato.com`
      - other tabs only if explicitly asked: `category:promotions`, `category:social`
      - read state: `is:unread`, `is:read`
      - subject:   `subject:invoice`   recency: `newer_than:7d`
    Combine with spaces, e.g. `from:linkedin.com is:unread`. user_id is auto-injected.
    """
    token = await resolve_user_token(user_id, "gmail")
    hdr = {"Authorization": f"Bearer {token}"}
    async with http_client() as c:
        lst = await c.get(f"{_BASE}/messages",
                          params={"q": query, "maxResults": max_results}, headers=hdr)
        lst.raise_for_status()
        messages = []
        for ref in lst.json().get("messages", []):
            mid = ref.get("id")
            if not mid:
                continue
            r = await c.get(f"{_BASE}/messages/{mid}",
                            params={"format": "full"}, headers=hdr)
            if r.status_code != 200:
                continue
            full = r.json()
            h = _headers_map(full.get("payload", {}))
            messages.append({
                "id": mid,
                "thread_id": full.get("threadId", ""),
                "from": h.get("from", ""),
                "subject": h.get("subject", ""),
                "date": h.get("date", ""),
                "snippet": full.get("snippet", ""),
                "body": _extract_body(full.get("payload", {}))[:_BODY_LIMIT],
            })
        log.info("gmail_searched", user_id=user_id, count=len(messages))
        return {"query": query, "messages": messages}


@mcp.tool()
async def search_contacts(user_id: str, query: str, max_results: int = 5) -> dict:
    """Look up Google contacts by name or partial — returns {name, email} candidates.

    Use this BEFORE send_email/create_draft when the user names a person without an
    email (e.g. "send John the report"). Requires the OAuth token to include the
    `https://www.googleapis.com/auth/contacts.readonly` scope.
    Args: user_id (auto-injected), query (name or partial), max_results.
    """
    token = await resolve_user_token(user_id, "gmail")
    async with http_client() as c:
        r = await c.get(_PEOPLE_SEARCH,
                        params={"query": query,
                                "readMask": "names,emailAddresses",
                                "pageSize": max_results},
                        headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        out = []
        for hit in r.json().get("results", []):
            person = hit.get("person", {})
            names = person.get("names", [])
            emails = person.get("emailAddresses", [])
            out.append({
                "name": names[0].get("displayName", "") if names else "",
                "email": emails[0].get("value", "") if emails else "",
            })
        log.info("gmail_contacts_searched", user_id=user_id, query=query, count=len(out))
        return {"query": query, "results": out}


@mcp.tool()
async def send_email(user_id: str, to: str, subject: str, body: str) -> dict:
    """Send an email via Gmail. `to` MUST be a valid email address.

    If the user only gave a name, FIRST call `search_contacts(query=<name>)` to
    resolve it to an address; only call `send_email` once you have a real email.
    Args: user_id (auto-injected), to, subject, body.
    """
    token = await resolve_user_token(user_id, "gmail")
    async with http_client() as c:
        r = await c.post(f"{_BASE}/messages/send",
                         json={"raw": _make_raw(to, subject, body)},
                         headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.info("gmail_email_sent", user_id=user_id, to=to)
        return r.json()


@mcp.tool()
async def create_draft(user_id: str, to: str, subject: str, body: str) -> dict:
    """Create a Gmail draft. Args: user_id (auto-injected), to, subject, body."""
    token = await resolve_user_token(user_id, "gmail")
    async with http_client() as c:
        r = await c.post(f"{_BASE}/drafts",
                         json={"message": {"raw": _make_raw(to, subject, body)}},
                         headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.info("gmail_draft_created", user_id=user_id)
        return r.json()


@mcp.tool()
async def reply_thread(user_id: str, thread_id: str, body: str) -> dict:
    """Reply to a Gmail thread. Args: user_id (auto-injected), thread_id, body."""
    token = await resolve_user_token(user_id, "gmail")
    async with http_client() as c:
        r = await c.post(f"{_BASE}/messages/send",
                         json={"raw": _make_raw("", "Re:", body), "threadId": thread_id},
                         headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        log.info("gmail_thread_replied", user_id=user_id, thread_id=thread_id)
        return r.json()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=7002, path="/mcp")
