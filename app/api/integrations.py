"""Integration token storage, listing, and disconnect endpoints."""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.data.repositories import delete_token, get_user_tokens, save_token
from app.security.crypto import pack_token, unpack_token

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1")

_ALLOWED_PROVIDERS = {"calendar", "gmail", "notion", "slack"}
# Access tokens within this many seconds of expiry are surfaced as "expiring".
_EXPIRING_WINDOW_SEC = 300


class TokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=1, description="Plain-text provider access token.")
    refresh_token: str | None = Field(default=None, description="OAuth refresh token for auto-renewal.")
    expires_in: int | None = Field(default=None, description="Access-token lifetime in seconds.")


def _require_user(request: Request) -> str:
    """Return the authenticated user_id or raise 401."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return user_id


def _token_status(token_enc: str) -> str:
    """Classify a stored token as valid | expiring | revoked from its expiry."""
    try:
        data = unpack_token(token_enc)
    except Exception:
        return "revoked"  # undecryptable / corrupt → treat as needing reconnection
    expires_at = data.get("expires_at")
    if not expires_at:
        return "valid"
    remaining = float(expires_at) - time.time()
    if remaining <= 0:
        return "revoked"
    return "expiring" if remaining <= _EXPIRING_WINDOW_SEC else "valid"


async def _revoke_upstream(provider: str, token_enc: str) -> None:
    """Best-effort revoke at the provider (mocked here; never fails the request)."""
    log.info("integration_upstream_revoke", provider=provider)


@router.get("/integrations")
async def list_integrations(request: Request) -> dict:
    """List the caller's connected providers and each token's status."""
    user_id = _require_user(request)
    rows = await get_user_tokens(user_id)
    integrations = [{"provider": r["provider"], "status": _token_status(r["token_enc"])}
                    for r in rows]
    log.debug("integrations_listed", user_id=user_id, count=len(integrations))
    return {"integrations": integrations}


@router.post("/integrations/{provider}/token")
async def store_integration_token(provider: str, req: TokenRequest, request: Request) -> dict:
    """Fernet-encrypt and upsert a provider token for the current user."""
    if provider not in _ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"unknown_provider:{provider}")
    user_id = _require_user(request)
    token_enc = pack_token(req.token, req.refresh_token, req.expires_in)
    await save_token(user_id=user_id, provider=provider, token_enc=token_enc)
    log.info("integration_token_stored", user_id=user_id, provider=provider)
    return {"status": "ok", "provider": provider}


@router.delete("/integrations/{provider}")
async def disconnect_integration(provider: str, request: Request) -> dict:
    """Delete the user's provider token row and revoke it upstream (best-effort)."""
    if provider not in _ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"unknown_provider:{provider}")
    user_id = _require_user(request)
    rows = await get_user_tokens(user_id)
    match = next((r for r in rows if r["provider"] == provider), None)
    if match is None or not await delete_token(user_id, provider):
        raise HTTPException(status_code=404, detail="integration_not_found")
    await _revoke_upstream(provider, match["token_enc"])
    log.info("integration_disconnected", user_id=user_id, provider=provider)
    return {"status": "disconnected", "provider": provider}
