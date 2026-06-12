"""Shared helpers for all MCP server files."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
import structlog
from fastapi import HTTPException

from app.config import get_settings
from app.data.repositories import get_token, save_token
from app.security.crypto import pack_token, unpack_token

log = structlog.get_logger(__name__)
_GOOGLE = {"gmail", "calendar"}
_GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"


@asynccontextmanager
async def http_client(timeout: int = 15):
    """Yield a short-lived httpx.AsyncClient with a sensible timeout."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        yield client


def _is_expired(expires_at: float | None, skew: int = 60) -> bool:
    """True if the token has expired or is within `skew` seconds of expiry."""
    return bool(expires_at) and time.time() + skew >= expires_at


async def _refresh_google_token(user_id: str, provider: str,
                                bundle: dict[str, Any]) -> str:
    """Exchange a refresh token for a fresh Google access token and persist it."""
    s = get_settings()
    log.info("google_token_refresh_start", user_id=user_id, provider=provider)
    async with http_client() as c:
        r = await c.post(_GOOGLE_TOKEN_URI, data={
            "client_id": s.gmail_client_id,
            "client_secret": s.gmail_client_secret,
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
