"""Shared helpers for all MCP server files."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
import structlog
from fastapi import HTTPException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.data.repositories import get_token, save_token
from app.security.crypto import pack_token, unpack_token

log = structlog.get_logger(__name__)
# "google" joins the existing gmail/calendar entries: the new combined google MCP stores its
# OAuth bundle under the "google" provider key and still needs the Google refresh path.
_GOOGLE = {"gmail", "calendar", "google"}
_GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"

# Transient network errors worth retrying; HTTP 4xx/5xx (raise_for_status) are NOT retried here
# because most are deterministic (bad symbol, auth) and the planner should see them promptly.
_RETRYABLE_EXC: tuple[type[BaseException], ...] = (httpx.TimeoutException, httpx.NetworkError)


@asynccontextmanager
async def http_client(timeout: int = 15):
    """Yield a short-lived httpx.AsyncClient with a sensible timeout."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        yield client


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=0.5, max=8),
       retry=retry_if_exception_type(_RETRYABLE_EXC), reraise=True)
async def get_json(client: httpx.AsyncClient, url: str, *,
                   params: dict[str, Any] | None = None,
                   headers: dict[str, str] | None = None) -> Any:
    """GET ``url`` with exponential-backoff retries on transient network errors; return JSON.

    Shared by the read-heavy MCP servers (finnhub, reddit). Simple exponential backoff only —
    no circuit breakers (CLAUDE.md). ``raise_for_status`` surfaces HTTP errors to the caller.
    """
    response = await client.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


def _is_expired(expires_at: float | None, skew: int = 60) -> bool:
    """True if the token has expired or is within `skew` seconds of expiry."""
    # Pre-existing mypy fix: narrow None/0.0 out first so the comparison is float-vs-float.
    if not expires_at:
        return False
    return time.time() + skew >= expires_at


async def _refresh_google_token(user_id: str, provider: str,
                                bundle: dict[str, Any]) -> str:
    """Exchange a refresh token for a fresh Google access token and persist it."""
    s = get_settings()
    # Prefer the new google_oauth_* creds; fall back to the original gmail_client_* so existing
    # .env files keep refreshing without change (the combined google server is Gmail + Calendar).
    client_id = s.google_oauth_client_id or s.gmail_client_id
    client_secret = s.google_oauth_client_secret or s.gmail_client_secret
    log.info("google_token_refresh_start", user_id=user_id, provider=provider)
    async with http_client() as c:
        r = await c.post(_GOOGLE_TOKEN_URI, data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": bundle["refresh_token"],
            "grant_type": "refresh_token",
        })
        if r.status_code != 200:
            log.error("google_token_refresh_failed", provider=provider,
                      status=r.status_code)
        r.raise_for_status()
        tok = r.json()
    access = tok["access_token"]
    await save_token(user_id, provider,
                     pack_token(access, bundle["refresh_token"], tok.get("expires_in")))
    log.info("google_token_refreshed", user_id=user_id, provider=provider)
    return access


async def resolve_user_token(user_id: str, provider: str) -> str:
    """Fetch + decrypt the integration token, refreshing expired Google tokens."""
    token_enc = await get_token(user_id, provider)
    if not token_enc:
        log.warning("missing_integration_token", user_id=user_id, provider=provider)
        raise HTTPException(status_code=403, detail=f"no_token:{provider}")
    bundle = unpack_token(token_enc)
    if (provider in _GOOGLE and bundle.get("refresh_token")
            and _is_expired(bundle.get("expires_at"))):
        return await _refresh_google_token(user_id, provider, bundle)
    return bundle["access_token"]
