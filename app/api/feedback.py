"""Feedback submission endpoint."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.data.repositories import save_feedback

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1")


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: int = Field(ge=1, le=5, description="User satisfaction score 1-5.")
    comment: str | None = Field(default=None, description="Optional free-text comment.")


@router.post("/feedback/{trace_id}")
async def submit_feedback(trace_id: str, req: FeedbackRequest, request: Request) -> dict:
    """Record user feedback score for a completed workflow run."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    await save_feedback(trace_id=trace_id, score=req.score, comment=req.comment)
    log.info("feedback_submitted", trace_id=trace_id, score=req.score, user_id=user_id)
    return {"status": "ok"}
