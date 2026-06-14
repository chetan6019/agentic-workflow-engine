"""POST /v1/invoke plus the phase/result polling endpoints the UI reads."""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.core.state import AgentState
from app.data.redis_client import get_redis
from app.data.repositories import create_session, get_plan_by_trace_id, save_plan
from app.orchestration.graph import compile_graph

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1")

# Hold strong refs to background runs so they aren't garbage-collected mid-await
# (asyncio only weakly references tasks; a dropped ref can cancel the run).
_BG_TASKS: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    """Run a coroutine in the background, keeping a reference until it finishes."""
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


class InvokeRequest(BaseModel):
    """Payload for starting a new workflow run."""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = Field(default=None, description="Existing session ID or None.")
    user_request: str = Field(min_length=4, description="Natural-language user request.")
    require_approval: bool = Field(
        default=False, description="If true, pause after planning and wait for /v1/approvals.")


class InvokeResponse(BaseModel):
    """Trace id; the UI polls /phase + /result for progress and the final state."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str = Field(description="Trace ID — poll /v1/invoke/phase/{id} and /result/{id}.")


_PHASE_KEY = "phase:{trace_id}"
_PHASE_TTL = 600


def _phase_label(state: AgentState) -> str:
    """Friendly current-phase label inferred from the accumulated AgentState."""
    if state.error:
        return "❌ failed"
    if state.requires_approval:
        return "🛑 awaiting approval"
    if state.draft and state.verdict is not None:
        return "💾 finalizing"
    if state.draft:
        return "🛡️ checking quality"
    if state.tool_results:
        return "✍️ composing reply"
    if state.plan:
        return "🔧 executing tools"
    if state.retrieved_plans:
        return "🧠 planning"
    return "📥 retrieving context"


async def _set_phase(trace_id: str, label: str, owner: str,
                     *, done: bool = False) -> None:
    """Write the current phase (+ owner) to Redis so the UI can poll it.

    ``owner`` lets the poll endpoint enforce ownership without a DB round-trip on
    every tick — the run's user_id travels with the phase key (REVIEW.md R15).
    """
    payload = json.dumps({"phase": label, "done": done, "owner": owner})
    await get_redis().set(_PHASE_KEY.format(trace_id=trace_id), payload, ex=_PHASE_TTL)


async def _run_workflow_with_phase(initial: AgentState) -> None:
    """Run the graph in the background, updating Redis phase + persisting the state."""
    compiled = compile_graph()
    config = {"configurable": {"thread_id": initial.trace_id}}
    owner = initial.user_id
    log.info("workflow_run_start", trace_id=initial.trace_id, user_id=owner)
    final_state = initial
    await _set_phase(initial.trace_id, "📥 retrieving context", owner)
    try:
        async for chunk in compiled.astream(initial, config=config, stream_mode="values"):
            try:
                final_state = AgentState.model_validate(chunk)
            except Exception:
                continue
            await _set_phase(initial.trace_id, _phase_label(final_state), owner)
    except Exception as exc:
        log.exception("workflow_run_failed", trace_id=initial.trace_id, error=str(exc))
        final_state.error = final_state.error or str(exc)
    try:
        await save_plan(final_state)
    except Exception as exc:
        log.warning("final_state_persist_failed", trace_id=initial.trace_id, error=str(exc))
    await _set_phase(initial.trace_id, _phase_label(final_state), owner, done=True)
    log.info("workflow_run_done", trace_id=initial.trace_id,
             has_draft=final_state.draft is not None, error=final_state.error)


@router.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest, request: Request) -> InvokeResponse:
    """Kick off the workflow in the background; UI polls /phase and /result."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")

    session_id = req.session_id or await create_session(user_id)
    trace_id = uuid4().hex
    log.info("invoke_accepted", trace_id=trace_id, user_id=user_id,
             session_id=str(session_id), new_session=req.session_id is None)

    initial = AgentState(
        trace_id=trace_id,
        user_id=user_id,
        session_id=str(session_id),
        user_request=req.user_request,
        requires_approval=req.require_approval,
    )
    await _set_phase(trace_id, "🚀 starting", user_id)
    _spawn(_run_workflow_with_phase(initial))
    return InvokeResponse(trace_id=trace_id)


def _require_user(request: Request) -> str:
    """Return the authenticated user_id or raise 401 (JWT middleware sets it)."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")
    return user_id


@router.get("/invoke/phase/{trace_id}")
async def invoke_phase(trace_id: str, request: Request) -> dict:
    """Return the current workflow phase for live progress, polled by the UI."""
    user_id = _require_user(request)
    raw = await get_redis().get(_PHASE_KEY.format(trace_id=trace_id))
    if not raw:
        return {"phase": "pending", "done": False}
    data = json.loads(raw)
    # Ownership: the run's user_id rides with the phase key, so a caller can only
    # see progress for their own runs (REVIEW.md R15). No DB hit on the poll path.
    if data.get("owner") and data["owner"] != user_id:
        raise HTTPException(status_code=404, detail="not_found")
    return {"phase": data.get("phase", "pending"), "done": data.get("done", False)}


@router.get("/invoke/result/{trace_id}")
async def invoke_result(trace_id: str, request: Request) -> dict:
    """Authoritative final state for a trace, read from the persisted plan.

    Requires a valid JWT and ownership of the run (REVIEW.md R15) — the final state
    can contain emails and calendar contents. The plan row is written mid-run (before
    the draft exists, to satisfy the tool_calls FK), so we mark it 'done' once a
    reliable terminal signal is present: an error, a populated draft, or a guardrails
    verdict (set only after the graph completes scoring).
    """
    user_id = _require_user(request)
    row = await get_plan_by_trace_id(trace_id)
    if not row:
        return {"status": "pending"}
    if row.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="not_found")
    state = json.loads(row["state_json"])
    draft_md = (state.get("draft") or {}).get("detail_markdown")
    done = bool(draft_md) or bool(state.get("error")) or state.get("verdict") is not None
    return {"status": "done" if done else "pending", "state": state}
