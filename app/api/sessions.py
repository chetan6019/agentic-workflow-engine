"""Session listing and detail routes."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.data.repositories import get_session, get_sessions_by_user, rename_session

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1")


class RenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200, description="New session title.")


@router.get("/sessions")
async def list_sessions(request: Request) -> dict:
    """Return all sessions for the authenticated user."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    sessions = await get_sessions_by_user(user_id)
    log.debug("sessions_listed", user_id=user_id, count=len(sessions))
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session_detail(session_id: str, request: Request) -> dict:
    """Return a single session and its messages for the authenticated user."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    session = await get_session(session_id)
    if not session or session.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="session_not_found")
    log.debug("session_fetched", session_id=session_id, user_id=user_id)
    return session


@router.patch("/sessions/{session_id}")
async def rename_session_endpoint(session_id: str, req: RenameRequest, request: Request) -> dict:
    """Rename a session (only the owner may rename it)."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    if not await rename_session(session_id, user_id, req.title.strip()):
        raise HTTPException(status_code=404, detail="session_not_found")
    return {"status": "ok", "title": req.title.strip()}
