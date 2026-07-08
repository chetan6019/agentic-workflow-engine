"""POST /v1/invoke plus the phase/result polling endpoints the UI reads."""

from __future__ import annotations

import asyncio
import json
import time
from uuid import uuid4

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.core.background import spawn
from app.core.metrics import workflow_run_latency_seconds
from app.core.state import AgentState
from app.data.redis_client import get_redis
from app.guardrails import evaluate_input
from app.data.repositories import (
    create_session,
    get_plan_by_trace_id,
    get_recent_messages,
    save_message,
    save_plan,
)
from app.orchestration.graph import compile_graph, discard_thread
from app.security.jwt_tokens import decode_access_token

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1")

# Runs execute as in-process asyncio tasks (via core.background.spawn, so shutdown
# can drain them) on the instance that accepted /invoke; they are NOT durable
# across a restart of that instance. Two guards bound the blast radius: a per-run
# timeout (run_timeout_sec) marks a hung run failed in process, and
# /invoke/result reconciles a run whose phase key has vanished (worker gone) to a
# failed state. Approval RESUME is restart-tolerant because the paused run's
# LangGraph checkpoint lives in Postgres (CHECKPOINTER_BACKEND=postgres) and
# Command(resume=...) can land on any instance.
# STALE (2026-07-07): previous wording — "because it rebuilds from Postgres" —
# described the retired route-to-END mechanism; superseded by native interrupt().


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
# Prior conversation turns fed to the planner for follow-up resolution. _history_block
# applies rolling compaction: the newest 4 turns render verbatim (200-char clip) and older
# turns compact to 80-char lines — so 8 turns (4 exchanges) costs barely more than the old
# 4-turn window while letting follow-ups reach further back.
_HISTORY_TURNS = 8
# Idempotency-Key → trace_id mapping TTL (replays within this window return the
# original run instead of starting a duplicate).
_IDEM_KEY = "idem:{user_id}:{key}"
_IDEM_TTL = 86_400
# SSE stream tuning: how often the server re-reads the phase key, and how often it
# sends a comment heartbeat so proxies don't drop an idle connection.
_SSE_POLL_SEC = 0.5
_SSE_HEARTBEAT_SEC = 15.0


def _sse_event(event: str, data: dict) -> str:
    """Render one server-sent event frame (event name + JSON data line)."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _phase_label(state: AgentState) -> str:
    """Friendly current-phase label inferred from the accumulated AgentState."""
    if state.error:
        return "❌ failed"
    if state.requires_approval:
        return "🛑 awaiting approval"
    # Greeting / direct-answer runs invoke no tools. Once the (empty) plan exists, collapse the
    # tool-oriented phases into a single "processing" pill so the UI never shows "executing
    # tools" / "composing reply" for a request that runs none. (The brief pre-plan retrieving/
    # planning phases are generic and apply to every run, so they're left as-is.)
    if state.plan is not None and not state.plan.steps:
        return "⏳ processing"
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


class _RunHolder:
    """Mutable view of a streaming run: latest state + whether it paused for HITL."""

    def __init__(self, state: AgentState) -> None:
        self.state = state
        self.paused = False


async def _run_workflow_with_phase(initial: AgentState) -> None:
    """Run the graph in the background, updating Redis phase + persisting the state."""
    compiled = compile_graph()
    config = {"configurable": {"thread_id": initial.trace_id}}
    owner = initial.user_id
    log.info("workflow_run_start", trace_id=initial.trace_id, user_id=owner)
    started = time.monotonic()
    # holder is mutable so a timeout still sees the latest state; "paused" flips when
    # the stream yields the __interrupt__ frame (a node called interrupt() for HITL).
    holder = _RunHolder(initial)

    async def _stream() -> None:
        async for chunk in compiled.astream(initial, config=config, stream_mode="values"):
            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                holder.paused = True
                continue
            try:
                holder.state = AgentState.model_validate(chunk)
            except Exception:
                continue
            await _set_phase(initial.trace_id, _phase_label(holder.state), owner)

    await _set_phase(initial.trace_id, "📥 retrieving context", owner)
    try:
        await asyncio.wait_for(_stream(), timeout=get_settings().run_timeout_sec)
    except asyncio.TimeoutError:
        log.error("workflow_run_timeout", trace_id=initial.trace_id,
                  timeout_sec=get_settings().run_timeout_sec)
        holder.state.error = holder.state.error or "run_timeout"
    except Exception as exc:
        log.exception("workflow_run_failed", trace_id=initial.trace_id, error=str(exc))
        holder.state.error = holder.state.error or str(exc)
    final_state = holder.state
    if holder.paused:
        # Paused for HITL. The interrupted node already persisted the approval draft +
        # token to Postgres itself; the last streamed state PREDATES the pause, so
        # saving it here would clobber that. Keep the checkpoint — it is the resume
        # source (Command(resume=...)) and, on the postgres backend, survives restarts.
        await _set_phase(initial.trace_id, "🛑 awaiting approval", owner, done=True)
        workflow_run_latency_seconds.labels(outcome="paused").observe(
            time.monotonic() - started)
        log.info("workflow_run_paused", trace_id=initial.trace_id)
        return
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
    # Terminal run (paused runs returned above): the checkpoint is no longer needed.
    await discard_thread(initial.trace_id)
    await _set_phase(initial.trace_id, _phase_label(final_state), owner, done=True)
    elapsed = time.monotonic() - started
    # Outcome label for the Prometheus histogram: a timed-out run is distinct from a
    # generic error so the dashboard can separate "slow" from "broken".
    if final_state.error == "run_timeout":
        outcome = "timeout"
    elif final_state.error:
        outcome = "error"
    else:
        outcome = "ok"
    workflow_run_latency_seconds.labels(outcome=outcome).observe(elapsed)
    log.info("workflow_run_done", trace_id=initial.trace_id,
             duration_ms=int(elapsed * 1000),
             has_draft=final_state.draft is not None, error=final_state.error)


@router.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest, request: Request) -> InvokeResponse:
    """Kick off the workflow in the background; UI polls /phase and /result.

    Input guardrails run synchronously here (the API boundary): a blocked request
    fails fast with 400, a rate-cap hit with 429, and PII is redacted before the
    planner ever sees it. An ``Idempotency-Key`` header replays the original run.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")

    idem_key = request.headers.get("Idempotency-Key")
    if idem_key:
        cached = await get_redis().get(_IDEM_KEY.format(user_id=user_id, key=idem_key))
        if cached:
            cached_trace = cached.decode() if isinstance(cached, (bytes, bytearray)) else cached
            log.info("invoke_idempotent_replay", trace_id=cached_trace, user_id=user_id)
            return InvokeResponse(trace_id=cached_trace)

    trace_id = uuid4().hex
    decision = await evaluate_input(req.user_request, user_id=user_id, trace_id=trace_id)
    if not decision.allowed:
        raise HTTPException(status_code=400, detail=f"input_guardrail:{','.join(decision.blocked_rules)}")
    if any(h.action == "rate_limit" for h in decision.hits):
        raise HTTPException(status_code=429, detail="rate_limited")
    vetted_request = decision.text  # PII-redacted prompt the planner will see

    session_id = str(req.session_id or await create_session(user_id))
    log.info("invoke_accepted", trace_id=trace_id, user_id=user_id,
             session_id=session_id, new_session=req.session_id is None)

    # Load prior turns BEFORE saving the current one, so history holds only the
    # conversation up to (not including) this request. Best-effort: a memory hiccup
    # must not block the run.
    try:
        history = await get_recent_messages(session_id, _HISTORY_TURNS)
        await save_message(session_id, "user", vetted_request)
    except Exception as exc:
        log.warning("history_load_failed", trace_id=trace_id, error=str(exc))
        history = []

    initial = AgentState(
        trace_id=trace_id,
        user_id=user_id,
        session_id=session_id,
        user_request=vetted_request,
        history=history,
        requires_approval=req.require_approval,
    )
    await _set_phase(trace_id, "🚀 starting", user_id)
    spawn(_run_workflow_with_phase(initial))
    if idem_key:
        await get_redis().set(_IDEM_KEY.format(user_id=user_id, key=idem_key), trace_id, ex=_IDEM_TTL)
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


async def _load_result(trace_id: str, user_id: str) -> dict:
    """Load a run's result dict — shared by /invoke/result and /invoke/stream.

    The plan row is written mid-run (before the draft exists, to satisfy the
    tool_calls FK), so the run counts as 'done' once a reliable terminal signal
    is present: an error, a populated draft, or a guardrails verdict (set only
    after the graph completes scoring).

    Reconciliation: if the row exists with no terminal signal AND the phase key
    has vanished, the worker that owned this run is gone (e.g. instance restart);
    report it failed rather than 'pending' forever.
    """
    row = await get_plan_by_trace_id(trace_id)
    if not row:
        return {"status": "pending"}
    if row.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="not_found")
    state = json.loads(row["state_json"])
    draft_md = (state.get("draft") or {}).get("detail_markdown")
    done = bool(draft_md) or bool(state.get("error")) or state.get("verdict") is not None
    if done:
        return {"status": "done", "state": state}
    if not await get_redis().get(_PHASE_KEY.format(trace_id=trace_id)):
        log.warning("run_reconciled_failed", trace_id=trace_id)
        state["error"] = state.get("error") or "instance_restarted"
        return {"status": "failed", "state": state}
    return {"status": "pending", "state": state}


@router.get("/invoke/result/{trace_id}")
async def invoke_result(trace_id: str, request: Request) -> dict:
    """Authoritative final state for a trace, read from the persisted plan.

    Requires a valid JWT and ownership of the run (REVIEW.md R15) — the final state
    can contain emails and calendar contents. Terminal/reconciliation rules live in
    _load_result, which the SSE stream endpoint shares.
    """
    return await _load_result(trace_id, _require_user(request))


@router.get("/invoke/stream/{trace_id}")
async def invoke_stream(trace_id: str, request: Request,
                        access_token: str | None = None) -> StreamingResponse:
    """SSE progress stream: `phase` events while the run advances, one `result` event, done.

    The browser's EventSource API cannot send an Authorization header, so the JWT
    may arrive as ``?access_token=`` instead — decoded with the same validator the
    auth middleware uses; header auth still works for curl/tests. Trade-off: the
    token shows up in the URL (and any access log). Acceptable here because tokens
    are short-lived and logs are local structlog — revisit via ADR if logs ever
    ship to a third party.

    The Streamlit UI keeps polling /invoke/phase + /invoke/result; this endpoint
    exists for the React UI (architecture v8: "Streaming is SSE").
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id and access_token:
        user_id = decode_access_token(access_token)
    if not user_id:
        raise HTTPException(status_code=401, detail="unauthenticated")

    # Already terminal (or the phase key expired after completion): reply with a
    # one-event stream instead of polling for a phase that will never change.
    # Raises the same 404 as /invoke/result when the run belongs to someone else.
    first = await _load_result(trace_id, user_id)

    async def _generate():
        if first["status"] != "pending":
            yield _sse_event("result", first)
            return
        deadline = time.monotonic() + get_settings().run_timeout_sec + 30
        last_phase: str | None = None
        last_beat = time.monotonic()
        while time.monotonic() < deadline:
            raw = await get_redis().get(_PHASE_KEY.format(trace_id=trace_id))
            if raw:
                data = json.loads(raw)
                # Same ownership rule as /invoke/phase: the owner rides in the key.
                if data.get("owner") and data["owner"] != user_id:
                    yield _sse_event("error", {"detail": "not_found"})
                    return
                phase = data.get("phase", "pending")
                if phase != last_phase:
                    last_phase = phase
                    last_beat = time.monotonic()
                    yield _sse_event("phase", {"phase": phase, "done": data.get("done", False)})
                if data.get("done"):
                    yield _sse_event("result", await _load_result(trace_id, user_id))
                    return
            if time.monotonic() - last_beat >= _SSE_HEARTBEAT_SEC:
                last_beat = time.monotonic()
                yield ": ping\n\n"  # comment frame keeps proxies from idling the stream out
            await asyncio.sleep(_SSE_POLL_SEC)
        yield _sse_event("error", {"detail": "stream_timeout"})

    # X-Accel-Buffering tells nginx (the React app's prod proxy) not to buffer SSE.
    return StreamingResponse(_generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
