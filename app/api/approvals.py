"""POST /v1/approvals/{token} — apply a HITL decision and resume the graph."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException, Request
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from app.core.state import AgentState, DraftResponse
from app.data.repositories import (
    get_approval_by_token,
    get_pending_approvals_by_user,
    get_plan_by_trace_id,
    save_plan,
    update_approval_status,
)
from app.orchestration.graph import compile_graph, discard_thread

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


# STALE (2026-07-07): moved to app/orchestration/nodes/orchestrator.py::_apply_rebook —
# with native interrupt() the rebook happens inside the paused node's resume path,
# not in the API layer. Kept per CLAUDE.md R13.
# def _rebook_calendar_conflict(state: AgentState) -> bool:
#     """On approval of a calendar clash, point the create_event step at the suggested slot.
#
#     Clears tool_results/draft/token so the resumed graph re-runs execute and books the
#     new time. Returns True if a rebookable conflict was found and applied.
#     """
#     for r in state.tool_results:
#         out = r.output or {}
#         if out.get("conflict") and out.get("suggested"):
#             sug = out["suggested"]
#             for step in (state.plan.steps if state.plan else []):
#                 if step.id == r.step_id:
#                     step.arguments["start"] = sug["start"]
#                     step.arguments["end"] = sug["end"]
#             state.tool_results = []
#             state.draft = None
#             state.approval_token = None
#             return True
#     return False


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


@router.get("/approvals")
async def list_approvals(request: Request, status: str = "pending") -> dict:
    """List the caller's pending approvals; expired ones auto-reject on read."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    if status != "pending":
        raise HTTPException(status_code=400, detail="unsupported_status")
    pending: list[dict] = []
    for row in await get_pending_approvals_by_user(user_id):
        if _is_expired(row.get("expires_at")):
            await update_approval_status(row["token"], "rejected")  # auto-reject stale
            await discard_thread(row["trace_id"])  # expired pause: free its checkpoint
            log.info("approval_auto_rejected", trace_id=row.get("trace_id"))
            continue
        pending.append({"token": row["token"], "trace_id": row["trace_id"],
                        "expires_at": str(row["expires_at"])})
    return {"approvals": pending}


@router.post("/approvals/{token}", response_model=ApprovalResponse)
async def submit_approval(token: str, req: ApprovalRequest, request: Request) -> ApprovalResponse:
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

    # Ownership (security fix): a JWT-authenticated caller may only act on their own
    # run. The {token} capability still authorizes anonymous (email-link) use, so the
    # check only fires when a caller identity is present and differs from the owner.
    caller = getattr(request.state, "user_id", None)
    owner = plan_row.get("user_id")
    if caller is not None and owner is not None and caller != owner:
        log.warning("approval_cross_user_denied", trace_id=approval.get("trace_id"))
        raise HTTPException(status_code=403, detail="forbidden")

    if req.decision == "edit" and req.edited_draft is None:
        raise HTTPException(status_code=400, detail="edited_draft_required")

    trace_id = approval["trace_id"]
    compiled = compile_graph()
    config = {"configurable": {"thread_id": trace_id}}

    # STALE (2026-07-07): resume used to rebuild AgentState from plans.state_json and
    # re-invoke the whole graph (the route-to-END pause mechanism). Superseded by native
    # interrupt()/Command(resume=...) — the durable checkpoint holds the paused state.
    # Kept per CLAUDE.md R13:
    # state = AgentState.model_validate_json(plan_row["state_json"])
    # if req.decision == "reject": ... save_plan; update_approval_status; return
    # if req.decision == "edit": state.draft = req.edited_draft
    # if req.decision == "approve" and _rebook_calendar_conflict(state): ...
    # state.requires_approval = False
    # final = await compiled.ainvoke(state, config=config)

    # A resumable thread must have a pending interrupted node in its checkpoint.
    # Missing checkpoint = restart on the memory backend or pre-migration approval:
    # nothing to resume — tell the caller instead of silently re-running from scratch.
    snapshot = await compiled.aget_state(config)
    if snapshot is None or not snapshot.next:
        log.warning("approval_not_resumable", trace_id=trace_id,
                    hint="checkpoint missing — is CHECKPOINTER_BACKEND=postgres set?")
        raise HTTPException(status_code=409, detail="approval_not_resumable")

    resume_payload = {
        "decision": req.decision,
        "edited_draft": req.edited_draft.model_dump() if req.edited_draft else None,
    }
    log.info("approval_resuming_graph", trace_id=trace_id, decision=req.decision)
    final = await compiled.ainvoke(Command(resume=resume_payload), config=config)
    final_state = AgentState.model_validate(final)

    await save_plan(final_state)
    # DB vocabulary: reject is recorded past-tense ("rejected"), matching the
    # auto-reject-on-expiry path; approve/edit are recorded as submitted.
    await update_approval_status(
        token, "rejected" if req.decision == "reject" else req.decision)
    await discard_thread(trace_id)  # resumed run is terminal either way (reject sets error)
    log.info("approval_resumed", trace_id=trace_id, decision=req.decision,
             confidence=round(final_state.confidence, 3), error=final_state.error)
    return ApprovalResponse(
        status="rejected" if req.decision == "reject" else "resumed")
