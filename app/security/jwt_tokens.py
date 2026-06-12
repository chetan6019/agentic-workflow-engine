"""JWT creation and validation using python-jose."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from jose import JWTError, jwt

from app.config import get_settings

_ALGORITHM = "HS256"
_EXPIRY_HOURS = 24


def create_access_token(user_id: str) -> str:
    """Create a signed HS256 JWT with a 24-hour expiry."""
    s = get_settings()
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=_EXPIRY_HOURS),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> str:
    """Decode and verify a JWT; return the user_id or raise HTTPException(401)."""
    s = get_settings()
    try:
        payload = jwt.decode(token, s.jwt_secret, algorithms=[_ALGORITHM])
        user_id: str | None = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="invalid_token")
        return user_id
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=f"token_error: {exc}") from exc
