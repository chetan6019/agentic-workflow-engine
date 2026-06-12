"""MCP Notion server — deterministic Notion adapter on port 7003."""

from __future__ import annotations

import structlog
from fastmcp import FastMCP

from app.mcp.servers._shared import http_client, resolve_user_token

log = structlog.get_logger(__name__)
mcp = FastMCP("notion")
_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json"}


@mcp.tool()
async def create_page(user_id: str, parent_id: str, title: str, content: str = "") -> dict:
    """Create a Notion page. Args: user_id (auto-injected), parent_id, title, content."""
    token = await resolve_user_token(user_id, "notion")
    children = ([{"object": "block", "type": "paragraph",
                  "paragraph": {"rich_text": [{"text": {"content": content}}]}}]
                if content else [])
    body = {"parent": {"page_id": parent_id},
            "properties": {"title": [{"text": {"content": title}}]},
            "children": children}
    async with http_client() as c:
        r = await c.post(f"{_BASE}/pages", json=body, headers=_headers(token))
        r.raise_for_status()
        log.info("notion_page_created", user_id=user_id)
        return r.json()


@mcp.tool()
async def append_block(user_id: str, block_id: str, text: str) -> dict:
    """Append a paragraph block. Args: user_id (auto-injected), block_id, text."""
    token = await resolve_user_token(user_id, "notion")
    body = {"children": [{"object": "block", "type": "paragraph",
                          "paragraph": {"rich_text": [{"text": {"content": text}}]}}]}
    async with http_client() as c:
        r = await c.patch(f"{_BASE}/blocks/{block_id}/children", json=body,
                          headers=_headers(token))
        r.raise_for_status()
        log.info("notion_block_appended", user_id=user_id)
        return r.json()


@mcp.tool()
async def search_pages(user_id: str, query: str, max_results: int = 10) -> dict:
    """Search Notion pages. Args: user_id (auto-injected), query, max_results."""
    token = await resolve_user_token(user_id, "notion")
    body = {"query": query, "page_size": max_results}
    async with http_client() as c:
        r = await c.post(f"{_BASE}/search", json=body, headers=_headers(token))
        r.raise_for_status()
        log.debug("notion_searched", user_id=user_id)
        return r.json()


@mcp.tool()
async def update_page(user_id: str, page_id: str, title: str | None = None,
                      archived: bool | None = None) -> dict:
    """Update a Notion page's title/archived. Args: user_id (auto-injected), page_id."""
    token = await resolve_user_token(user_id, "notion")
    patch: dict = {}
    if title is not None:
        patch["properties"] = {"title": [{"text": {"content": title}}]}
    if archived is not None:
        patch["archived"] = archived
    async with http_client() as c:
        r = await c.patch(f"{_BASE}/pages/{page_id}", json=patch, headers=_headers(token))
        r.raise_for_status()
        log.info("notion_page_updated", user_id=user_id)
        return r.json()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=7003, path="/mcp")
