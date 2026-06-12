"""Workflow orchestrator node: phase dispatch, tool fan-out, persistence."""

from __future__ import annotations

import asyncio
from typing import Literal

import structlog

import datetime

from app.agents.response_composer import compose
from app.core.state import AgentState, DraftResponse, ExecutionPlan, PlanStep, ToolResult
from app.data.repositories import create_approval, get_session, save_plan, save_tool_call
from app.mcp.client import get_mcp_client
from app.rag.indexer import index_plan
from app.rag.retriever import retrieve

log = structlog.get_logger(__name__)

Phase = Literal["entry", "execute", "finalize", "noop"]

# Deletions always need human approval BEFORE they run, gated pre-execution here so we
# never approve a removal that already happened. Updates are NOT pre-gated: an update is
# only paused when its new time clashes with another event, which the calendar server
# detects at execution time and routes through the same conflict path as a create.
_APPROVAL_ACTIONS = {"delete_event", "delete_page"}


def _needs_approval(steps: list[PlanStep]) -> bool:
    """True if any plan step deletes an existing resource."""
    return any(s.action in _APPROVAL_ACTIONS for s in steps)


def _conflict_result(results: list[ToolResult]) -> ToolResult | None:
    """Return the first calendar-conflict tool result (event not created), if any."""
    for r in results:
        if r.ok and isinstance(r.output, dict) and r.output.get("conflict"):
            return r
    return None


def _fmt_local(ts: str) -> str:
    """Render an RFC3339 timestamp as a friendly local time, e.g. 'Mon 15 Jun 2026, 11:00 IST'."""
    try:
        return datetime.datetime.fromisoformat(ts).strftime("%a %d %b %Y, %H:%M") + " IST"
    except ValueError:
        return ts


def _fmt_day(ts: str) -> str:
    """Render an RFC3339 timestamp as a date only, e.g. 'Fri 13 Jun 2026'."""
    try:
        return datetime.datetime.fromisoformat(ts).strftime("%a %d %b %Y")
    except ValueError:
        return ts


def _approval_draft(plan: ExecutionPlan) -> DraftResponse:
    """Deterministic approval draft listing the changes awaiting human approval.

    Built from plan-step arguments only (no LLM), so the user sees exactly what they
    are approving before any tool runs — at this point no composed draft exists yet.
    """
    items: list[str] = []
    for s in plan.steps:
        if s.action not in _APPROVAL_ACTIONS:
            continue
        a = s.arguments
        verb = "delete" if s.action.startswith("delete_") else "update"
        if s.action.endswith("_event"):
            what = a.get("match_summary") or a.get("summary") or a.get("event_id") or "an event"
            when = a.get("time_min") or a.get("start")
            target = f'the calendar event "{what}"' + (f" on {_fmt_day(when)}" if when else "")
        elif s.action.endswith("_page"):
            target = f'the Notion page "{a.get("title") or a.get("page_id") or "a page"}"'
        else:
            target = s.action.replace("_", " ")
        items.append(f"{verb} {target}")
    joined = ", and ".join(items) or "the requested change"
    return DraftResponse(
        summary=f"Approval needed to {joined}.",
        detail_markdown=(f"⚠️ This will {joined}.\n\n"
                         f"**Approve** to proceed, or **Reject** to cancel."),
        actions_taken=[], actions_pending=[f"Awaiting approval to {joined}"], citations=[])


def _conflict_draft(conflict: ToolResult) -> DraftResponse:
    """Deterministic approval draft asking the user to confirm the suggested free slot."""
    out = conflict.output or {}
    req = out.get("requested", {})
    suggested = out.get("suggested")
    clashes = ", ".join(c.get("summary", "(busy)") for c in out.get("conflicting", []))
    req_when = _fmt_local(req.get("start", ""))
    if suggested:
        slot = _fmt_local(suggested["start"])
        summary = f"That slot ({req_when}) is taken — next free slot is {slot}."
        detail = (f"⚠️ **{req_when}** clashes with **{clashes}**.\n\n"
                  f"The next free 30-minute working-hours slot is **{slot}**.\n\n"
                  f"**Approve** to book it instead, or **Reject** to cancel.")
        pending = [f"Awaiting approval to book at {slot}"]
    else:
        summary = f"That slot ({req_when}) is taken and no free slot was found in the next 7 days."
        detail = (f"⚠️ **{req_when}** clashes with **{clashes}**, and I couldn't find a free "
                  f"working-hours slot in the next 7 days. Try a different time.")
        pending = ["No free slot found in the next 7 days"]
    return DraftResponse(summary=summary, detail_markdown=detail,
                         actions_taken=[], actions_pending=pending, citations=[])


def _phase(state: AgentState) -> Phase:
    """Decide which phase to run based on which state fields are populated."""
    if state.plan is None and not state.retrieved_plans:
        return "entry"
    if state.plan is not None and not state.tool_results:
        return "execute"
    if state.draft is not None and state.confidence > 0.0:
        return "finalize"
    return "noop"


def _topo_levels(steps: list[PlanStep]) -> list[list[PlanStep]]:
    """Return Kahn-style topological levels for parallel execution."""
    remaining = {s.id: set(s.depends_on) for s in steps}
    by_id = {s.id: s for s in steps}
    levels: list[list[PlanStep]] = []
    while remaining:
        ready = [sid for sid, deps in remaining.items() if not deps]
        if not ready:
            raise ValueError("cycle_in_plan_steps")
        levels.append([by_id[sid] for sid in ready])
        for sid in ready:
            remaining.pop(sid)
        for deps in remaining.values():
            deps.difference_update(ready)
    return levels


async def _execute_level(level: list[PlanStep], mcp, parallel: bool,
                         *, trace_id: str, user_id: str) -> list[ToolResult]:
    """Run one DAG level either in parallel or sequentially.

    MCP tools take flat parameters; ``user_id`` is injected for token resolution,
    and ``step_id``/``trace_id`` are metadata kept out of the tool input.
    """
    coros = [
        mcp.call_tool(
            step.tool, step.action,
            {**step.arguments, "user_id": user_id},
            trace_id=trace_id, step_id=step.id,
        )
        for step in level
    ]
    if parallel:
        return list(await asyncio.gather(*coros, return_exceptions=False))
    results: list[ToolResult] = []
    for c in coros:
        results.append(await c)
    return results


async def _run_entry(state: AgentState) -> AgentState:
    """Load session context and seed retrieved plans."""
    await get_session(state.session_id)
    state.retrieved_plans = await retrieve(state)
    log.info("orchestrator_entry_done", retrieved=len(state.retrieved_plans))
    return state


async def _run_execute(state: AgentState) -> AgentState:
    """Persist the plan, fan out tool calls in DAG order, then compose the draft."""
    assert state.plan is not None
    # Update/delete steps mutate an existing resource and need approval BEFORE execution.
    # Only flag when no token exists yet: once approved (token set, requires_approval
    # cleared on resume) we fall through and execute instead of pausing again.
    if _needs_approval(state.plan.steps) and not state.approval_token:
        state.requires_approval = True
    # HITL gate: if the user (or planner) requested approval and we haven't
    # already issued a token, pause here BEFORE any tool runs.
    if state.requires_approval and not state.approval_token:
        await save_plan(state)  # plans row must exist before the approvals FK insert
        state.approval_token = await create_approval(state.trace_id)
        # Give the approval card something to show: describe the pending change(s).
        # On resume the real composed draft replaces this after the tools run.
        if state.draft is None and _needs_approval(state.plan.steps):
            state.draft = _approval_draft(state.plan)
        await save_plan(state)  # persist the issued token so resume can find it
        log.info("orchestrator_awaiting_approval", trace_id=state.trace_id,
                 token=state.approval_token[:8] + "…")
        return state
    await save_plan(state)  # create the plans row so tool_calls FK is satisfied
    mcp = get_mcp_client()
    parallel = state.plan.strategy in {"parallel", "mixed"}
    levels = _topo_levels(state.plan.steps)
    log.info("orchestrator_execute_start", steps=len(state.plan.steps),
             levels=len(levels), strategy=state.plan.strategy)
    for i, level in enumerate(levels):
        results = await _execute_level(level, mcp, parallel,
                                       trace_id=state.trace_id, user_id=state.user_id)
        for r in results:
            state.tool_results.append(r)
            await save_tool_call(state.trace_id, r)
        log.debug("orchestrator_level_done", level=i,
                  ok=sum(1 for r in results if r.ok), total=len(results))
    # Calendar clash: the event was NOT created. If a free slot was found, pause for
    # HITL approval so the user can confirm booking it; the graph routes to END here.
    conflict = _conflict_result(state.tool_results)
    if conflict is not None:
        state.draft = _conflict_draft(conflict)
        if (conflict.output or {}).get("suggested") and not state.approval_token:
            state.approval_token = await create_approval(state.trace_id)
            state.requires_approval = True
            log.info("orchestrator_calendar_conflict_pause", trace_id=state.trace_id,
                     token=state.approval_token[:8] + "…")
        await save_plan(state)
        return state
    state.draft = await compose(state)
    await save_plan(state)  # persist the draft immediately so the result survives
    log.info("orchestrator_execute_done", tool_results=len(state.tool_results),
             failed=sum(1 for r in state.tool_results if not r.ok))
    return state


async def _run_finalize(state: AgentState) -> AgentState:
    """Persist final plan and schedule background indexing."""
    await save_plan(state)
    asyncio.create_task(index_plan(state))
    log.info("orchestrator_finalized", confidence=round(state.confidence, 3))
    return state


async def orchestrator_node(state: AgentState) -> AgentState:
    """Dispatch to the correct phase handler based on current state."""
    structlog.contextvars.bind_contextvars(trace_id=state.trace_id, user_id=state.user_id)
    phase = _phase(state)
    log.info("orchestrator_node", phase=phase)
    try:
        if phase == "entry":
            return await _run_entry(state)
        if phase == "execute":
            return await _run_execute(state)
        if phase == "finalize":
            return await _run_finalize(state)
        return state
    except Exception as exc:
        log.exception("orchestrator_failed", phase=phase, error=str(exc))
        state.error = f"orchestrator_failed:{exc!s}"
        return state
