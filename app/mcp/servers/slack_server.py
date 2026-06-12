"""MCP Slack server — deterministic Slack Web API adapter on port 7004."""

from __future__ import annotations

import structlog
from fastmcp import FastMCP

from app.mcp.servers._shared import http_client, resolve_user_token

log = structlog.get_logger(__name__)
mcp = FastMCP("slack")
_BASE = "https://slack.com/api"


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


@mcp.tool()
async def send_message(user_id: str, channel: str, text: str) -> dict:
    """Post a message to a Slack channel. Args: user_id (auto-injected), channel, text."""
    token = await resolve_user_token(user_id, "slack")
    async with http_client() as c:
        r = await c.post(f"{_BASE}/chat.postMessage",
                         json={"channel": channel, "text": text}, headers=_headers(token))
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "slack_error"))
        log.info("slack_message_sent", user_id=user_id, channel=channel)
        return data


@mcp.tool()
async def schedule_message(user_id: str, channel: str, text: str, post_at: int) -> dict:
    """Schedule a Slack message. Args: user_id (auto-injected), channel, text, post_at (unix ts)."""
    token = await resolve_user_token(user_id, "slack")
    async with http_client() as c:
        r = await c.post(f"{_BASE}/chat.scheduleMessage",
                         json={"channel": channel, "text": text, "post_at": post_at},
                         headers=_headers(token))
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "slack_error"))
        log.info("slack_message_scheduled", user_id=user_id, channel=channel)
        return data


@mcp.tool()
async def search_messages(user_id: str, query: str, count: int = 20) -> dict:
    """Search Slack messages. Args: user_id (auto-injected), query, count."""
    token = await resolve_user_token(user_id, "slack")
    async with http_client() as c:
        r = await c.get(f"{_BASE}/search.messages",
                        params={"query": query, "count": count}, headers=_headers(token))
        r.raise_for_status()
        data = r.json()
        log.debug("slack_searched", user_id=user_id)
        return data


@mcp.tool()
async def list_channels(user_id: str, limit: int = 100) -> dict:
    """List Slack channels. Args: user_id (auto-injected), limit."""
    token = await resolve_user_token(user_id, "slack")
    async with http_client() as c:
        r = await c.get(f"{_BASE}/conversations.list",
                        params={"limit": limit}, headers=_headers(token))
        r.raise_for_status()
        data = r.json()
        log.debug("slack_channels_listed", user_id=user_id)
        return data


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=7004, path="/mcp")
