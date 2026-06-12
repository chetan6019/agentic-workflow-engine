"""Integration token storage endpoint."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.data.repositories import save_token
from app.security.crypto import pack_token

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1")

_ALLOWED_PROVIDERS = {"calendar", "gmail", "notion", "slack"}


class TokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=1, description="Plain-text provider access token.")
    refresh_token: str | None = Field(default=None, description="OAuth refresh token for auto-renewal.")
    expires_in: int | None = Field(default=None, description="Access-token lifetime in seconds.")


@router.post("/integrations/{provider}/token")
async def store_integration_token(provider: str, req: TokenRequest, request: Request) -> dict:
    """Fernet-encrypt and upsert a provider token for the current user."""
    if provider not in _ALLOWED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"unknown_provider:{provider}")
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    token_enc = pack_token(req.token, req.refresh_token, req.expires_in)
    await save_token(user_id=user_id, provider=provider, token_enc=token_enc)
    log.info("integration_token_stored", user_id=user_id, provider=provider)
    return {"status": "ok", "provider": provider}
