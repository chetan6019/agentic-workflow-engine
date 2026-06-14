"""POST /v1/invoke plus the phase/result polling endpoints the UI uses."""

from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.core.background import spawn
from app.core.state import AgentState
from app.data.redis_client import get_redis
from app.data.repositories import (
    create_session,
    get_plan_by_trace_id,
    get_recent_messages,
    save_message,
    save_plan,
)
from app.orchestration.graph import compile_graph, discard_thread

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1")

# Runs execute as in-process asyncio tasks (via core.background.spawn, so shutdown
# can drain them) on the instance that accepted /invoke; they are NOT durable
# across a restart of that instance. Two guards bound the blast radius: a per-run
# timeout (run_timeout_sec) marks a hung run failed in process, and
# /invoke/result reconciles a run whose phase key has vanished (worker gone) to a
# failed state. Approval RESUME is restart-tolerant because it rebuilds from
# Postgres and can land on any instance.


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
# Prior conversation turns fed to the planner for follow-up resolution.
_HISTORY_TURNS = 6


def _phase_label(state: AgentState) -> str:
    """Friendly current-phase label inferred from the accumulated AgentState."""
    if state.error:
        return "❌ failed"
    if state.requires_approval:
        return "🛑 awaiting approval"
    if state.draft and state.confidence > 0:
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


async def _set_phase(trace_id: str, label: str, *, done: bool = False) -> None:
    """Write the current phase to Redis so the UI can poll it."""
    payload = json.dumps({"phase": label, "done": done})
    await get_redis().set(_PHASE_KEY.format(trace_id=trace_id), payload, ex=_PHASE_TTL)


async def _run_workflow_with_phase(initial: AgentState) -> None:
    """Run the graph in the background, updating Redis phase + persisting the state."""
    compiled = compile_graph()
    config = {"configurable": {"thread_id": initial.trace_id}}
    log.info("workflow_run_start", trace_id=initial.trace_id, user_id=initial.user_id)
    started = time.monotonic()
    holder = {"state": initial}  # mutable so a timeout still sees the latest state

    async def _stream() -> None:
        async for chunk in compiled.astream(initial, config=config, stream_mode="values"):
            try:
                holder["state"] = AgentState.model_validate(chunk)
            except Exception:
                continue
            await _set_phase(initial.trace_id, _phase_label(holder["state"]))

    await _set_phase(initial.trace_id, "📥 retrieving context")
    try:
        await asyncio.wait_for(_stream(), timeout=get_settings().run_timeout_sec)
    except asyncio.TimeoutError:
        log.error("workflow_run_timeout", trace_id=initial.trace_id,
                  timeout_sec=get_settings().run_timeout_sec)
        holder["state"].error = holder["state"].error or "run_timeout"
    except Exception as exc:
        log.exception("workflow_run_failed", trace_id=initial.trace_id, error=str(exc))
        holder["state"].error = holder["state"].error or str(exc)
    final_state = holder["state"]
    try:
        await save_plan(final_state)
    except Exception as exc:
        log.warning("final_state_persist_failed", trace_id=initial.trace_id, error=str(exc))
    # Record the assistant turn so the next request's planner has it as context.
    if final_state.draft is not None:
        try:
            await save_message(initial.session_id, "assistant", final_state.draft.summary)
        except Exception as exc:
            log.warning("assistant_message_persist_failed",
                        trace_id=initial.trace_id, error=str(exc))
    # State is now in Postgres; drop the in-memory checkpoint (resume rebuilds
    # from Postgres, so it's never read again — see discard_thread).
    await discard_thread(initial.trace_id)
    await _set_phase(initial.trace_id, _phase_label(final_state), done=True)
    log.info("workflow_run_done", trace_id=initial.trace_id,
             duration_ms=int((time.monotonic() - started) * 1000),
             has_draft=final_state.draft is not None, error=final_state.error)


@router.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest, request: Request) -> InvokeResponse:
    """Kick off the workflow in the background; UI polls /phase and /result."""
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")

    session_id = str(req.session_id or await create_session(user_id))
    trace_id = uuid4().hex
    log.info("invoke_accepted", trace_id=trace_id, user_id=user_id,
             session_id=session_id, new_session=req.session_id is None)

    # Load prior turns BEFORE saving the current one, so history holds only the
    # conversation up to (not including) this request. Best-effort: a memory hiccup
    # must not block the run.
    try:
        history = await get_recent_messages(session_id, _HISTORY_TURNS)
        await save_message(session_id, "user", req.user_request)
    except Exception as exc:
        log.warning("history_load_failed", trace_id=trace_id, error=str(exc))
        history = []

    initial = AgentState(
        trace_id=trace_id,
        user_id=user_id,
        session_id=session_id,
        user_request=req.user_request,
        history=history,
        requires_approval=req.require_approval,
    )
    await _set_phase(trace_id, "🚀 starting")
    spawn(_run_workflow_with_phase(initial))
    return InvokeResponse(trace_id=trace_id)


@router.get("/invoke/phase/{trace_id}")
async def invoke_phase(trace_id: str) -> dict:
    """Return the current workflow phase for live progress, polled by the UI."""
    raw = await get_redis().get(_PHASE_KEY.format(trace_id=trace_id))
    if not raw:
        return {"phase": "pending", "done": False}
    return json.loads(raw)


@router.get("/invoke/result/{trace_id}")
async def invoke_result(trace_id: str) -> dict:
    """Authoritative final state for a trace, read from the persisted plan.

    The UI polls this for the final answer. The plan row is written mid-run
    (before the draft exists, to satisfy the tool_calls FK), so we mark it 'done'
    once any reliable terminal signal is present: an error, a populated draft, or
    a non-zero confidence (which only gets set after guardrails runs, i.e. after
    the graph fully completes).

    Reconciliation: if the row exists with no terminal signal AND the phase key
    has vanished, the worker that owned this run is gone (e.g. instance restart);
    report it failed rather than 'pending' forever.
    """
    row = await get_plan_by_trace_id(trace_id)
    if not row:
        return {"status": "pending"}
    state = json.loads(row["state_json"])
    draft_md = (state.get("draft") or {}).get("detail_markdown")
    done = bool(draft_md) or bool(state.get("error")) or float(state.get("confidence") or 0) > 0
    if done:
        return {"status": "done", "state": state}
    if not await get_redis().get(_PHASE_KEY.format(trace_id=trace_id)):
        log.warning("run_reconciled_failed", trace_id=trace_id)
        state["error"] = state.get("error") or "instance_restarted"
        return {"status": "failed", "state": state}
    return {"status": "pending", "state": state}
