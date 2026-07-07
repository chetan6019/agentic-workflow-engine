"""MCP Finnhub server — READ-ONLY market-data adapter on port 7005.

API key only (no per-user OAuth). Read-only BY DESIGN: there is no place_order / trade / buy /
sell tool here, and those actions are also denied in app/guardrails/policy.yaml as defense in
depth. A per-minute Redis counter keeps us under Finnhub's free-tier quota.
"""

from __future__ import annotations

import time

import structlog
from fastmcp import FastMCP

from app.config import get_settings
from app.data.redis_client import get_redis
from app.mcp.servers._shared import get_json, http_client

log = structlog.get_logger(__name__)
mcp = FastMCP("finnhub")
_BASE = "https://finnhub.io/api/v1"
# Matches policy.yaml's finnhub.rate_per_minute (kept in sync by hand; the cap is enforced here,
# not in the output guardrails, because a call-rate cap is not an output-text concern).
_RATE_PER_MINUTE = 60


def _token() -> str:
    """Return the Finnhub API key, or raise a clear error when it isn't configured."""
    key = get_settings().finnhub_api_key
    if not key:
        raise RuntimeError("finnhub_api_key_not_configured")
    return key


async def _within_rate_limit() -> bool:
    """True while the global Finnhub call rate is under the per-minute cap (fail-open on Redis)."""
    try:
        redis = get_redis()
        bucket = int(time.time() // 60)
        key = f"finnhub:rate:{bucket}"
        count = int(await redis.incrby(key, 1))
        if count == 1:  # first hit in this minute — set the TTL so the bucket self-expires
            await redis.expire(key, 60)
        return count <= _RATE_PER_MINUTE
    except Exception as exc:  # a cap must never lock the tool out on a Redis hiccup
        log.warning("finnhub_rate_check_failed", error=str(exc))
        return True


async def _get(path: str, params: dict[str, object]) -> object:
    """Rate-checked GET against the Finnhub REST API; injects the API token."""
    if not await _within_rate_limit():
        raise RuntimeError("finnhub_rate_limited")
    query = {**params, "token": _token()}
    async with http_client() as c:
        return await get_json(c, f"{_BASE}{path}", params=query)


@mcp.tool()
async def get_quote(symbol: str) -> dict:
    """Get the latest quote for a stock symbol (e.g. "AAPL"). Returns current/open/high/low."""
    data = await _get("/quote", {"symbol": symbol.upper()})
    log.info("finnhub_quote", symbol=symbol)
    return {"symbol": symbol.upper(), "quote": data}


@mcp.tool()
async def get_company_profile(symbol: str) -> dict:
    """Get the company profile (name, industry, market cap) for a stock symbol."""
    data = await _get("/stock/profile2", {"symbol": symbol.upper()})
    log.info("finnhub_profile", symbol=symbol)
    return {"symbol": symbol.upper(), "profile": data}


@mcp.tool()
async def get_company_news(symbol: str, from_date: str, to_date: str) -> dict:
    """Get recent company news. Args: symbol, from_date, to_date (both YYYY-MM-DD)."""
    data = await _get("/company-news",
                      {"symbol": symbol.upper(), "from": from_date, "to": to_date})
    items = data if isinstance(data, list) else []
    log.info("finnhub_news", symbol=symbol, count=len(items))
    return {"symbol": symbol.upper(), "news": items}


@mcp.tool()
async def search_symbol(query: str) -> dict:
    """Search for ticker symbols matching a company name or partial symbol."""
    data = await _get("/search", {"q": query})
    log.info("finnhub_search", query=query)
    return {"query": query, "results": data}


@mcp.tool()
async def get_basic_financials(symbol: str) -> dict:
    """Get basic financial metrics (margins, ratios, 52-week range) for a stock symbol."""
    data = await _get("/stock/metric", {"symbol": symbol.upper(), "metric": "all"})
    log.info("finnhub_financials", symbol=symbol)
    return {"symbol": symbol.upper(), "financials": data}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=7005, path="/mcp")
