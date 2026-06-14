"""POST /v1/approvals/{token} — apply a HITL decision and resume the graph."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.core.state import AgentState, DraftResponse
from app.data.repositories import (
    get_approval_by_token,
    get_plan_by_trace_id,
    save_plan,
    update_approval_status,
)
from app.orchestration.graph import compile_graph

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1")


class ApprovalRequest(BaseModel):
    """Decision payload for a HITL approval."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "edit", "reject"] = Field(description="User's decision.")
    edited_draft: DraftResponse | None = Field(
        default=None, description="Replacement draft when decision is 'edit'."
    )


class ApprovalResponse(BaseModel):
    """Status returned after handling the approval."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["resumed", "rejected"] = Field(description="Resulting status.")


def _rebook_calendar_conflict(state: AgentState) -> bool:
    """On approval of a calendar clash, point the create_event step at the suggested slot.

    Clears tool_results/draft/token so the resumed graph re-runs execute and books the
    new time. Returns True if a rebookable conflict was found and applied.
    """
    for r in state.tool_results:
        out = r.output or {}
        if out.get("conflict") and out.get("suggested"):
            sug = out["suggested"]
            for step in (state.plan.steps if state.plan else []):
                if step.id == r.step_id:
                    step.arguments["start"] = sug["start"]
                    step.arguments["end"] = sug["end"]
            state.tool_results = []
            state.draft = None
            state.approval_token = None
            return True
    return False


def _is_expired(expires_at) -> bool:
    """Return True if the expires_at timestamp is in the past."""
    if expires_at is None:
        return False
    if isinstance(expires_at, str):
        try:
            expires_at = datetime.fromisoformat(expires_at)
        except ValueError:
            return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at < datetime.now(timezone.utc)


@router.post("/approvals/{token}", response_model=ApprovalResponse)
async def submit_approval(token: str, req: ApprovalRequest) -> ApprovalResponse:
    """Apply the user's decision to the paused workflow and resume the graph."""
    log.info("approval_received", token=token[:8] + "…", decision=req.decision)
    approval = await get_approval_by_token(token)
    if not approval:
        log.warning("approval_not_found", token=token[:8] + "…")
        raise HTTPException(status_code=404, detail="approval_not_found")
    if _is_expired(approval.get("expires_at")):
        log.warning("approval_expired", trace_id=approval.get("trace_id"))
        raise HTTPException(status_code=410, detail="approval_expired")

    plan_row = await get_plan_by_trace_id(approval["trace_id"])
    if not plan_row:
        log.warning("approval_plan_not_found", trace_id=approval.get("trace_id"))
        raise HTTPException(status_code=404, detail="plan_not_found")

    # state_json is stored as a JSON string (model_dump_json), so parse, don't model_validate.
    state = AgentState.model_validate_json(plan_row["state_json"])

    if req.decision == "reject":
        state.error = "rejected_by_user"
        state.requires_approval = False
        await save_plan(state)
        await update_approval_status(token, "rejected")
        log.info("approval_rejected", trace_id=state.trace_id)
        return ApprovalResponse(status="rejected")

    if req.decision == "edit":
        if req.edited_draft is None:
            raise HTTPException(status_code=400, detail="edited_draft_required")
        state.draft = req.edited_draft

    if req.decision == "approve" and _rebook_calendar_conflict(state):
        log.info("approval_calendar_rebook", trace_id=state.trace_id)

    state.requires_approval = False

    compiled = compile_graph()
    config = {"configurable": {"thread_id": state.trace_id}}
    log.info("approval_resuming_graph", trace_id=state.trace_id, decision=req.decision)
    final = await compiled.ainvoke(state, config=config)
    final_state = AgentState.model_validate(final)

    await save_plan(final_state)
    await update_approval_status(token, req.decision)
    log.info("approval_resumed", trace_id=state.trace_id,
             confidence=round(final_state.confidence, 3), error=final_state.error)
    return ApprovalResponse(status="resumed")
