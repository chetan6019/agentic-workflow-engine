"""Fernet symmetric encryption for provider tokens."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Any

from cryptography.fernet import Fernet

from app.config import get_settings


@lru_cache(maxsize=1)
def get_fernet() -> Fernet:
    """Return the cached Fernet cipher initialised from FERNET_KEY."""
    key = get_settings().fernet_key
    if not key:
        key = Fernet.generate_key().decode()
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt(text: str) -> str:
    """Encrypt a plain-text string and return the ciphertext as a str."""
    return get_fernet().encrypt(text.encode()).decode()


def decrypt(token_enc: str) -> str:
    """Decrypt a Fernet ciphertext and return the plain-text string."""
    return get_fernet().decrypt(token_enc.encode()).decode()


def pack_token(access_token: str, refresh_token: str | None = None,
               expires_in: int | None = None) -> str:
    """Serialise an access/refresh token bundle as encrypted JSON."""
    payload = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": time.time() + expires_in if expires_in else None,
    }
    return encrypt(json.dumps(payload))


def unpack_token(token_enc: str) -> dict[str, Any]:
    """Decrypt a token bundle, falling back to a legacy raw access token."""
    raw = decrypt(token_enc)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        data = None
    if isinstance(data, dict) and "access_token" in data:
        return data
    return {"access_token": raw, "refresh_token": None, "expires_at": None}
