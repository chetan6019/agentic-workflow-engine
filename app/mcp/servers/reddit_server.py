"""MCP Reddit server — read-heavy adapter on port 7006.

Reads use an app-only (client_credentials) OAuth token cached in Redis. The two write tools
(post_comment, submit) need a user-context token (password grant) and are ALWAYS HITL-gated in
the guardrails regardless of confidence — posting can damage reputation. Reddit reads are
deliberately NOT in policy.yaml's trusted_read_actions: public content is untrusted by nature,
so the output content-scan stays ON for every read here.
"""

from __future__ import annotations

import base64

import structlog
from fastmcp import FastMCP

from app.config import get_settings
from app.data.redis_client import get_redis
from app.mcp.servers._shared import get_json, http_client

log = structlog.get_logger(__name__)
mcp = FastMCP("reddit")
_TOKEN_URI = "https://www.reddit.com/api/v1/access_token"
_API = "https://oauth.reddit.com"
# Reddit requires a descriptive, unique User-Agent or it rate-limits/blocks the client.
_USER_AGENT = "agentic-workflow-engine/1.0 (mcp reddit server)"
_APP_TOKEN_KEY = "reddit:app_token"


def _basic_auth(client_id: str, client_secret: str) -> str:
    """Build the HTTP Basic auth header value Reddit's token endpoint expects."""
    raw = f"{client_id}:{client_secret}".encode()
    return "Basic " + base64.b64encode(raw).decode()


async def _app_token() -> str:
    """Return a cached app-only Reddit token, fetching + caching a fresh one when absent."""
    s = get_settings()
    if not (s.reddit_client_id and s.reddit_client_secret):
        raise RuntimeError("reddit_client_credentials_not_configured")
    redis = get_redis()
    try:
        cached = await redis.get(_APP_TOKEN_KEY)
        if cached:
            return str(cached)
    except Exception as exc:  # cache is an optimisation, not a hard dependency
        log.warning("reddit_token_cache_read_failed", error=str(exc))
    async with http_client() as c:
        r = await c.post(_TOKEN_URI, data={"grant_type": "client_credentials"},
                         headers={"Authorization": _basic_auth(s.reddit_client_id, s.reddit_client_secret),
                                  "User-Agent": _USER_AGENT})
        r.raise_for_status()
        tok = r.json()
    token = str(tok["access_token"])
    try:  # cache just shy of the real expiry so we never serve an expired token
        ttl = max(int(tok.get("expires_in", 3600)) - 60, 60)
        await redis.set(_APP_TOKEN_KEY, token, ex=ttl)
    except Exception as exc:
        log.warning("reddit_token_cache_write_failed", error=str(exc))
    return token


async def _user_token() -> str:
    """Return a user-context token (password grant) for the HITL-gated write tools."""
    s = get_settings()
    if not (s.reddit_client_id and s.reddit_client_secret
            and s.reddit_username and s.reddit_password):
        raise RuntimeError("reddit_user_credentials_not_configured")
    async with http_client() as c:
        r = await c.post(_TOKEN_URI,
                         data={"grant_type": "password", "username": s.reddit_username,
                               "password": s.reddit_password},
                         headers={"Authorization": _basic_auth(s.reddit_client_id, s.reddit_client_secret),
                                  "User-Agent": _USER_AGENT})
        r.raise_for_status()
        return str(r.json()["access_token"])


async def _read(path: str, params: dict[str, object]) -> object:
    """Authenticated app-only GET against the Reddit OAuth API."""
    token = await _app_token()
    async with http_client() as c:
        return await get_json(c, f"{_API}{path}", params=params,
                              headers={"Authorization": f"Bearer {token}",
                                       "User-Agent": _USER_AGENT})


@mcp.tool()
async def search_subreddit(subreddit: str, query: str, limit: int = 10) -> dict:
    """Search within a subreddit. Args: subreddit (no 'r/' prefix), query, limit."""
    data = await _read(f"/r/{subreddit}/search",
                       {"q": query, "restrict_sr": 1, "limit": limit})
    log.info("reddit_search", subreddit=subreddit, query=query)
    return {"subreddit": subreddit, "query": query, "results": data}


@mcp.tool()
async def get_post(post_id: str) -> dict:
    """Fetch a single post by its base-36 id (without the 't3_' prefix)."""
    data = await _read(f"/by_id/t3_{post_id}", {})
    log.info("reddit_get_post", post_id=post_id)
    return {"post_id": post_id, "post": data}


@mcp.tool()
async def list_top(subreddit: str, limit: int = 10, time_window: str = "day") -> dict:
    """List top posts in a subreddit. Args: subreddit, limit, time_window (hour/day/week/...)."""
    data = await _read(f"/r/{subreddit}/top", {"t": time_window, "limit": limit})
    log.info("reddit_list_top", subreddit=subreddit)
    return {"subreddit": subreddit, "top": data}


@mcp.tool()
async def list_user_history(username: str, limit: int = 10) -> dict:
    """List a user's recent public posts/comments. Args: username (no 'u/' prefix), limit."""
    data = await _read(f"/user/{username}/overview", {"limit": limit})
    log.info("reddit_user_history", username=username)
    return {"username": username, "history": data}


@mcp.tool()
async def post_comment(parent_id: str, text: str) -> dict:
    """Reply to a post/comment. HITL-GATED (reputation risk). Args: parent_id (t3_/t1_), text."""
    token = await _user_token()
    async with http_client() as c:
        r = await c.post(f"{_API}/api/comment",
                         data={"thing_id": parent_id, "text": text, "api_type": "json"},
                         headers={"Authorization": f"Bearer {token}", "User-Agent": _USER_AGENT})
        r.raise_for_status()
        log.info("reddit_comment_posted", parent_id=parent_id)
        return r.json()


@mcp.tool()
async def submit(subreddit: str, title: str, text: str) -> dict:
    """Submit a self (text) post to a subreddit. HITL-GATED. Args: subreddit, title, text."""
    token = await _user_token()
    async with http_client() as c:
        r = await c.post(f"{_API}/api/submit",
                         data={"sr": subreddit, "title": title, "text": text,
                               "kind": "self", "api_type": "json"},
                         headers={"Authorization": f"Bearer {token}", "User-Agent": _USER_AGENT})
        r.raise_for_status()
        log.info("reddit_submitted", subreddit=subreddit)
        return r.json()


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=7006, path="/mcp")
