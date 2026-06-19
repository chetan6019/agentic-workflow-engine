"""Feedback submission endpoint."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.data.redis_client import check_rate_limit
from app.data.repositories import get_plan_by_trace_id, save_feedback

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1")

# Max feedback submissions per user per minute (cheap abuse guard, fail-open).
_FEEDBACK_LIMIT_PER_MIN = 20


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score: int = Field(ge=1, le=5, description="User satisfaction score 1-5.")
    comment: str | None = Field(default=None, description="Optional free-text comment.")


@router.post("/feedback/{trace_id}")
async def submit_feedback(trace_id: str, req: FeedbackRequest, request: Request) -> dict:
    """Record user feedback for one of the caller's own completed runs."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    if not await check_rate_limit(f"feedback:{user_id}", _FEEDBACK_LIMIT_PER_MIN):
        log.warning("feedback_rate_limited", user_id=user_id)
        raise HTTPException(status_code=429, detail="rate_limit_exceeded")
    # Ownership: feedback may only be attached to a run the caller owns.
    plan_row = await get_plan_by_trace_id(trace_id)
    if not plan_row:
        raise HTTPException(status_code=404, detail="run_not_found")
    if plan_row.get("user_id") != user_id:
        log.warning("feedback_cross_user_denied", trace_id=trace_id, user_id=user_id)
        raise HTTPException(status_code=403, detail="forbidden")
    await save_feedback(trace_id=trace_id, score=req.score, comment=req.comment)
    log.info("feedback_submitted", trace_id=trace_id, score=req.score, user_id=user_id)
    return {"status": "ok"}
