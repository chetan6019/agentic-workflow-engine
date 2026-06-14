"""JWT creation and validation using python-jose."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


def decode_access_token(token: str) -> str | None:
    """Decode and verify a JWT; return the user_id, or None if invalid/expired.

    The security layer returns a domain value rather than raising a web error —
    callers (the JWT middleware) translate a None into the right HTTP response.
    """
    s = get_settings()
    try:
        payload = jwt.decode(token, s.jwt_secret, algorithms=[_ALGORITHM])
    except JWTError:
        return None
    user_id = payload.get("sub")
    return user_id if isinstance(user_id, str) and user_id else None
