"""Execute node: pre-execution approval gate, DAG fan-out to MCP, HITL pauses (v9 stage 4)."""

from __future__ import annotations

import asyncio
import datetime
import time
from zoneinfo import ZoneInfo

import structlog
from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from app.config import get_settings
from app.core.metrics import node_latency_seconds
from app.core.state import AgentState, DraftResponse, ExecutionPlan, PlanStep, ToolResult
from app.data.repositories import get_or_create_approval, save_plan, save_tool_call
from app.guardrails import (
    approval_required_actions,
    confidence_gate_threshold,
    trusted_read_actions,
)
from app.mcp.client import get_mcp_client

log = structlog.get_logger(__name__)

# Which planner confidence levels each policy threshold gates.
_GATED_CONFIDENCE = {"medium": {"medium", "low"}, "low": {"low"}, "off": set()}


def _steps_needing_approval(steps: list[PlanStep]) -> dict[str, str]:
    """Map step_id -> reason ("policy" | "low_confidence") for steps that must pause.

    A step pauses BEFORE execution when its tool.action is in policy.yaml's
    approval_required_actions (any confidence), OR when the planner rated it
    medium/low (per confidence_gate_threshold) AND it is not a trusted read —
    reads on trusted sources never pause on confidence alone (rule 6 carve-out).
    """
    gated = approval_required_actions()
    trusted = trusted_read_actions()
    levels = _GATED_CONFIDENCE.get(confidence_gate_threshold(), {"medium", "low"})
    reasons: dict[str, str] = {}
    for s in steps:
        key = f"{s.tool}.{s.action}"
        if key in gated:
            reasons[s.id] = "policy"
        elif s.confidence in levels and key not in trusted:
            reasons[s.id] = "low_confidence"
    if reasons:
        # debug, not info: callers may evaluate this more than once per pause.
        log.debug("pre_execution_gate", reasons=reasons)
    return reasons


def _needs_approval(steps: list[PlanStep]) -> bool:
    """True if any plan step must pause for approval before execution."""
    return bool(_steps_needing_approval(steps))


def _conflict_result(results: list[ToolResult]) -> ToolResult | None:
    """Return the first calendar-conflict tool result (event not created), if any."""
    for r in results:
        if r.ok and isinstance(r.output, dict) and r.output.get("conflict"):
            return r
    return None


def _tz_label() -> str:
    """Short zone abbreviation (e.g. 'IST') for the configured default timezone."""
    tz = get_settings().default_tz
    try:
        return datetime.datetime.now(ZoneInfo(tz)).strftime("%Z") or tz
    except Exception:
        return tz


def _fmt_local(ts: str) -> str:
    """Render an RFC3339 timestamp as a friendly local time, e.g. 'Mon 15 Jun 2026, 11:00 IST'."""
    try:
        return datetime.datetime.fromisoformat(ts).strftime("%a %d %b %Y, %H:%M") + f" {_tz_label()}"
    except ValueError:
        return ts


def _fmt_day(ts: str) -> str:
    """Render an RFC3339 timestamp as a date only, e.g. 'Fri 13 Jun 2026'."""
    try:
        return datetime.datetime.fromisoformat(ts).strftime("%a %d %b %Y")
    except ValueError:
        return ts


def _approval_draft(plan: ExecutionPlan,
                    reasons: dict[str, str] | None = None) -> DraftResponse:
    """Deterministic approval draft listing the changes awaiting human approval.

    Built from plan-step arguments only (no LLM), so the user sees exactly what they
    are approving before any tool runs — at this point no composed draft exists yet.
    ``reasons`` is _steps_needing_approval's map; when omitted it is recomputed.
    """
    if reasons is None:
        reasons = _steps_needing_approval(plan.steps)
    items: list[str] = []
    low_conf_steps: list[PlanStep] = []
    for s in plan.steps:
        if s.id not in reasons:
            continue
        if reasons[s.id] == "low_confidence":
            low_conf_steps.append(s)
        a = s.arguments
        if s.action == "send_email":
            subj = a.get("subject")
            target = f"an email to {a.get('to') or 'a recipient'}"
            items.append(f"send {target}" + (f' (subject: "{subj}")' if subj else ""))
            continue
        verb = "delete" if s.action.startswith("delete_") else "update"
        if s.action.endswith("_event"):
            what = a.get("match_summary") or a.get("summary") or a.get("event_id") or "an event"
            when = a.get("time_min") or a.get("start")
            target = f'the calendar event "{what}"' + (f" on {_fmt_day(when)}" if when else "")
        elif s.action.endswith("_page"):
            target = f'the Notion page "{a.get("title") or a.get("page_id") or "a page"}"'
        else:
            # Generic wording for the other gated writes (reddit.submit,
            # github.close_pr, ...): "reddit: submit" beats "update submit".
            items.append(f"{s.tool}: {s.action.replace('_', ' ')}")
            continue
        items.append(f"{verb} {target}")
    joined = ", and ".join(items) or "the requested change"
    detail = f"⚠️ This will {joined}.\n\n"
    if low_conf_steps:
        lines = "\n".join(
            f"- `{s.tool}.{s.action}` — the planner was not fully sure this is what "
            f"you meant ({s.confidence} confidence)" for s in low_conf_steps)
        detail += ("**Why am I being asked?** One or more steps were rated below "
                   f"high confidence:\n{lines}\n\n")
    detail += "**Approve** to proceed, or **Reject** to cancel."
    return DraftResponse(
        summary=f"Approval needed to {joined}.",
        detail_markdown=detail,
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


async def _pause_for_approval(state: AgentState, draft: DraftResponse, kind: str) -> dict:
    """Persist the approval context, then interrupt() the graph until a human decides.

    First pass: interrupt() RAISES, the run pauses, and the checkpoint holds this
    node's input. The draft/token written to Postgres here is what the UI shows.
    Resume: LangGraph REPLAYS this node from the top, this function runs again
    (side effects must be — and are — idempotent: save_plan upserts,
    get_or_create_approval never mints twice), and interrupt() RETURNS the
    Command(resume=...) payload: {"decision": ..., "edited_draft": ...}.
    """
    state.requires_approval = True
    if state.draft is None:
        state.draft = draft
    await save_plan(state)  # plans row must exist before the approvals FK insert
    state.approval_token = await get_or_create_approval(state.trace_id)
    await save_plan(state)  # persist token + draft so the approval card can render
    log.info("execute_awaiting_approval", trace_id=state.trace_id, kind=kind,
             token=state.approval_token[:8] + "…")
    resume = interrupt({"kind": kind, "token": state.approval_token,
                        "draft": draft.model_dump()})
    return resume if isinstance(resume, dict) else {"decision": str(resume)}


def _apply_rebook(state: AgentState, conflict: ToolResult) -> None:
    """Point the clashing calendar step at the approved suggested slot.

    Clears tool_results/draft so the execute loop re-runs and books the new time.
    """
    sug = (conflict.output or {}).get("suggested") or {}
    for step in (state.plan.steps if state.plan else []):
        if step.id == conflict.step_id:
            step.arguments["start"] = sug.get("start")
            step.arguments["end"] = sug.get("end")
    state.tool_results = []
    state.draft = None


async def _run(state: AgentState) -> AgentState:
    """Gate the plan, then fan out tool calls in DAG order.

    Both HITL pauses (plan-stage gate, calendar-conflict rebook) use LangGraph's
    native interrupt(); POST /v1/approvals/{token} resumes with Command(resume=...).
    """
    assert state.plan is not None
    # ── Plan-stage gate: pause BEFORE any tool runs (policy writes, low planner
    # confidence, or an explicit require_approval request). plan_approved guards
    # the replan loop and interrupt replays from re-pausing an approved plan.
    gated = _steps_needing_approval(state.plan.steps)
    if (gated or state.requires_approval) and not state.plan_approved:
        resume = await _pause_for_approval(
            state, _approval_draft(state.plan, gated), kind="plan_approval")
        decision = resume.get("decision")
        if decision == "reject":
            state.error = "rejected_by_user"
            state.requires_approval = False
            await save_plan(state)
            return state
        if decision == "edit" and resume.get("edited_draft"):
            state.draft = DraftResponse.model_validate(resume["edited_draft"])
        state.requires_approval = False
        state.plan_approved = True
    await save_plan(state)  # create the plans row so tool_calls FK is satisfied
    mcp = get_mcp_client()
    parallel = state.plan.strategy in {"parallel", "mixed"}
    # Execute levels in a loop so an approved calendar rebook can clear results
    # and run again with the new slot. Tool idempotency keys (R8) make any
    # replayed step safe — unchanged inputs are deduplicated by the MCP client.
    while True:
        levels = _topo_levels(state.plan.steps)
        log.info("execute_start", steps=len(state.plan.steps),
                 levels=len(levels), strategy=state.plan.strategy)
        for i, level in enumerate(levels):
            results = await _execute_level(level, mcp, parallel,
                                           trace_id=state.trace_id, user_id=state.user_id)
            for r in results:
                state.tool_results.append(r)
                await save_tool_call(state.trace_id, r)
            log.debug("execute_level_done", level=i,
                      ok=sum(1 for r in results if r.ok), total=len(results))
        # Calendar clash: the event was NOT created.
        conflict = _conflict_result(state.tool_results)
        if conflict is None:
            break
        if not (conflict.output or {}).get("suggested"):
            # No free slot found: report it as the final answer — the compose node
            # skips a non-None draft, so this conflict draft reaches guard as-is.
            state.draft = _conflict_draft(conflict)
            await save_plan(state)
            return state
        # A free slot was suggested — pause for the user to confirm booking it.
        resume = await _pause_for_approval(
            state, _conflict_draft(conflict), kind="calendar_conflict")
        if resume.get("decision") == "reject":
            state.error = "rejected_by_user"
            state.requires_approval = False
            await save_plan(state)
            return state
        state.requires_approval = False
        _apply_rebook(state, conflict)  # clears results/draft; loop re-executes
        log.info("execute_calendar_rebook", trace_id=state.trace_id)
    # Drop any leftover approval-card draft: the compose node only composes when
    # state.draft is None, and the real answer must supersede the approval card.
    state.draft = None
    log.info("execute_done", tool_results=len(state.tool_results),
             failed=sum(1 for r in state.tool_results if not r.ok))
    return state


async def execute_node(state: AgentState) -> AgentState:
    """LangGraph node: approval gate + DAG tool fan-out, with HITL interrupt() pauses."""
    with structlog.contextvars.bound_contextvars(
        trace_id=state.trace_id,
        user_id=state.user_id,
        session_id=state.session_id,
        node="execute",
    ):
        t0 = time.monotonic()
        try:
            result = await _run(state)
        except GraphInterrupt:
            # interrupt() pauses the run for HITL — LangGraph's machinery must see
            # this, so it is NOT an execute failure. Re-raise untouched.
            raise
        except Exception as exc:
            log.exception("execute_failed", error=str(exc))
            state.error = f"execute_failed:{exc!s}"
            result = state
        node_latency_seconds.labels(node="execute").observe(time.monotonic() - t0)
        return result
