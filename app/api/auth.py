"""Authentication routes: register, login, me."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.data.repositories import (
    create_user,
    get_user_by_id,
    get_user_by_username,
    get_user_providers,
)
from app.security.jwt_tokens import create_access_token
from app.security.passwords import hash_password, verify_password

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1/auth")


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=3)
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    password: str


@router.post("/register")
async def register(req: RegisterRequest) -> dict:
    """Register a new user; returns user_id and access_token."""
    if await get_user_by_username(req.username):
        raise HTTPException(status_code=409, detail="username_taken")
    user = await create_user(req.username, hash_password(req.password))
    token = create_access_token(user["id"])
    log.info("user_registered", user_id=user["id"])
    return {"user_id": user["id"], "access_token": token}


@router.post("/login")
async def login(req: LoginRequest) -> dict:
    """Verify credentials and return a JWT access token."""
    user = await get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid_credentials")
    token = create_access_token(user["id"])
    log.info("user_login", user_id=user["id"])
    return {"access_token": token}


@router.get("/me")
async def me(request: Request) -> dict:
    """Return the currently authenticated user's profile."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="user_not_found")
    providers = await get_user_providers(user_id)
    user["integrations"] = {p: True for p in providers}
    log.debug("me_called", user_id=user_id)
    return user
