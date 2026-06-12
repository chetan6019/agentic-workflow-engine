"""Routing-band tests for the LangGraph conditional edges, plus DAG levelling.

These drive ``_route_after_orchestrator`` / ``_route_after_guard`` as plain
functions — no LLM, no DB, no compiled graph — so every band of the guardrails
routing table (finalize / HITL / re-plan / block) is pinned down cheaply.
"""

from __future__ import annotations

import pytest
from langgraph.graph import END

from app.core.state import AgentState, DraftResponse, ExecutionPlan, PlanStep, ToolResult
from app.orchestration import graph
from app.orchestration.nodes.orchestrator import _topo_levels


def _plan(steps: list[PlanStep] | None = None) -> ExecutionPlan:
    """Minimal valid ExecutionPlan."""
    return ExecutionPlan(reasoning="r", steps=steps or [], strategy="sequential",
                         complexity_score=1, estimated_cost_usd=0.0, requires_approval=False)


def _draft() -> DraftResponse:
    """Minimal valid DraftResponse."""
    return DraftResponse(summary="s", detail_markdown="d",
                         actions_taken=[], actions_pending=[])


def _state(**overrides) -> AgentState:
    """AgentState with required fields filled and overrides applied."""
    base: dict = dict(trace_id="t", user_id="u", session_id="s", user_request="do a thing")
    base.update(overrides)
    return AgentState(**base)


# ── _route_after_orchestrator ────────────────────────────────────────────────


def test_orchestrator_error_routes_to_end():
    assert graph._route_after_orchestrator(_state(error="boom")) == END


def test_orchestrator_without_plan_routes_to_planner():
    assert graph._route_after_orchestrator(_state()) == "planner"


def test_orchestrator_hitl_pause_routes_to_end():
    s = _state(plan=_plan(), requires_approval=True, approval_token="tok")
    assert graph._route_after_orchestrator(s) == END


def test_orchestrator_unscored_draft_routes_to_guardrails():
    s = _state(plan=_plan(), draft=_draft())  # confidence defaults to 0.0
    assert graph._route_after_orchestrator(s) == "guardrails"


def test_orchestrator_scored_draft_routes_to_end():
    s = _state(plan=_plan(), draft=_draft(), confidence=0.9)
    assert graph._route_after_orchestrator(s) == END


# ── _route_after_guard ───────────────────────────────────────────────────────


def test_guard_error_routes_to_end():
    assert graph._route_after_guard(_state(error="boom", confidence=0.9)) == END


def test_guard_high_confidence_finalizes():
    assert graph._route_after_guard(_state(confidence=0.9)) == "orchestrator"


def test_guard_medium_band_finalizes_while_hitl_disabled():
    """0.55–0.85 auto-finalizes while _HITL_ENABLED is False (current default)."""
    s = _state(confidence=0.7)
    assert graph._route_after_guard(s) == "orchestrator"
    assert s.requires_approval is False


def test_guard_medium_band_pauses_when_hitl_enabled(monkeypatch):
    """Flipping _HITL_ENABLED makes the medium band pause for approval."""
    monkeypatch.setattr(graph, "_HITL_ENABLED", True)
    s = _state(confidence=0.7)
    assert graph._route_after_guard(s) == END
    assert s.requires_approval is True


def test_guard_boundary_085_finalizes_even_with_hitl(monkeypatch):
    """conf == 0.85 is outside the HITL band — always finalizes."""
    monkeypatch.setattr(graph, "_HITL_ENABLED", True)
    assert graph._route_after_guard(_state(confidence=0.85)) == "orchestrator"


def test_guard_boundary_055_is_not_a_replan():
    assert graph._route_after_guard(_state(confidence=0.55)) == "orchestrator"


def test_guard_low_confidence_replans_and_resets_state():
    s = _state(
        confidence=0.4, retry_count=0, plan=_plan(), draft=_draft(),
        tool_results=[ToolResult(step_id="s1", ok=True, output={}, error=None, latency_ms=1)],
    )
    assert graph._route_after_guard(s) == "planner"
    assert s.retry_count == 1
    assert s.plan is None
    assert s.tool_results == []
    assert s.draft is None
    assert s.confidence == 0.0


def test_guard_retry_exhaustion_blocks_with_explanation():
    s = _state(confidence=0.4, retry_count=2)
    assert graph._route_after_guard(s) == END
    assert s.error == "low_confidence_blocked"


# ── _topo_levels (orchestrator DAG execution order) ──────────────────────────


def _step(sid: str, deps: list[str] | None = None) -> PlanStep:
    return PlanStep(id=sid, tool="gmail", action="send_email", depends_on=deps or [])


def test_topo_levels_orders_by_dependencies():
    levels = _topo_levels([_step("a"), _step("b", ["a"]), _step("c", ["a"])])
    assert [s.id for s in levels[0]] == ["a"]
    assert {s.id for s in levels[1]} == {"b", "c"}
    assert len(levels) == 2


def test_topo_levels_raises_on_cycle():
    with pytest.raises(ValueError, match="cycle_in_plan_steps"):
        _topo_levels([_step("a", ["b"]), _step("b", ["a"])])
