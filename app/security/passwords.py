"""bcrypt password hashing and verification."""

from __future__ import annotations

import bcrypt


def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt cost 12."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored bcrypt hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())
