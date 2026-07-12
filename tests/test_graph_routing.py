"""Routing tests for the v9 explicit-node LangGraph edges, plus DAG levelling.

The verdict bands live in ``guard.decide_verdict`` (which mutates state) and the graph
routers are pure readers of ``state.error`` / ``state.verdict`` (REVIEW.md R6). These
tests cover three layers: the per-stage error routers, ``decide_verdict`` for the
finalize / HITL / re-plan / block decision + its state changes, and the execute-node
gate helpers. No LLM, DB, or graph invocation involved.
"""

from __future__ import annotations

import pytest
from langgraph.graph import END

from app.core.state import AgentState, DraftResponse, ExecutionPlan, PlanStep, ToolResult
from app.orchestration import graph
from app.orchestration.nodes import execute, guard
from app.orchestration.nodes.execute import _approval_draft, _needs_approval, _topo_levels


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


# ── per-stage error routers (v9: any node that sets state.error goes to END) ─
# STALE (2026-07-12): the _route_after_orchestrator tests were retired with the
# phase-dispatch orchestrator node — the explicit-node routers below replace them.


@pytest.mark.parametrize(("router", "next_node"), [
    (graph._route_after_intake, "retrieve"),
    (graph._route_after_retrieve, "plan"),
    (graph._route_after_plan, "execute"),
    (graph._route_after_execute, "compose"),
    (graph._route_after_compose, "guard"),
])
def test_stage_router_advances_or_ends_on_error(router, next_node):
    assert router(_state()) == next_node
    assert router(_state(error="boom")) == END


def test_compiled_graph_has_the_seven_v9_nodes():
    # The topology is the spec (docs/langgraph_topology_v9.dot): one node per stage.
    compiled = graph._build_graph().compile()
    expected = {"intake", "retrieve", "plan", "execute", "compose", "guard", "respond"}
    assert expected.issubset(set(compiled.get_graph().nodes))


# ── _route_after_guard (pure verdict → destination mapping) ───────────────────


def test_guard_error_routes_to_end():
    assert graph._route_after_guard(_state(error="boom", verdict="finalize")) == END


def test_guard_finalize_routes_to_respond():
    assert graph._route_after_guard(_state(verdict="finalize")) == "respond"


def test_guard_replan_routes_to_plan_skipping_retrieve():
    assert graph._route_after_guard(_state(verdict="replan")) == "plan"


def test_guard_hitl_routes_to_end():
    assert graph._route_after_guard(_state(verdict="hitl")) == END


def test_guard_block_routes_to_end():
    assert graph._route_after_guard(_state(verdict="block", error="low_confidence_blocked")) == END


# ── guard._decide_verdict (the routing bands + their state changes, R6) ────────


def test_decide_high_confidence_finalizes():
    s = _state(confidence=0.9)
    guard.decide_verdict(s)
    assert s.verdict == "finalize"


def test_decide_medium_band_finalizes_while_hitl_disabled():
    """0.55–0.85 auto-finalizes while HITL_ENABLED is False (current default)."""
    s = _state(confidence=0.7)
    guard.decide_verdict(s)
    assert s.verdict == "finalize"
    assert s.requires_approval is False


def test_decide_medium_band_hitl_when_enabled(monkeypatch):
    """Flipping _HITL_ENABLED makes the medium band ask for approval."""
    monkeypatch.setattr(guard, "HITL_ENABLED", True)
    s = _state(confidence=0.7)
    guard.decide_verdict(s)
    assert s.verdict == "hitl"
    assert s.requires_approval is True


def test_decide_boundary_085_finalizes_even_with_hitl(monkeypatch):
    """conf == 0.85 is outside the HITL band — always finalizes."""
    monkeypatch.setattr(guard, "HITL_ENABLED", True)
    s = _state(confidence=0.85)
    guard.decide_verdict(s)
    assert s.verdict == "finalize"


def test_decide_boundary_055_is_not_a_replan():
    s = _state(confidence=0.55)
    guard.decide_verdict(s)
    assert s.verdict == "finalize"


def test_decide_low_confidence_replans_and_resets_state():
    s = _state(
        confidence=0.4, retry_count=0, plan=_plan(), draft=_draft(),
        tool_results=[ToolResult(step_id="s1", ok=True, output={}, error=None, latency_ms=1)],
    )
    guard.decide_verdict(s)
    assert s.verdict == "replan"
    assert s.retry_count == 1
    assert s.plan is None
    assert s.tool_results == []
    assert s.draft is None
    assert s.confidence == 0.0


def test_decide_retry_exhaustion_blocks_with_explanation():
    s = _state(confidence=0.4, retry_count=2)
    guard.decide_verdict(s)
    assert s.verdict == "block"
    assert s.error == "low_confidence_blocked"


def test_decide_noops_when_error_already_set():
    s = _state(error="pii_detected", confidence=0.0)
    guard.decide_verdict(s)
    assert s.verdict is None  # router sends error states straight to END


# ── _topo_levels (execute-node DAG execution order) ──────────────────────────


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


# ── _needs_approval: the pre-execution HITL gate ─────────────────────────────
# Regression for the "approve after the write already ran" gap: every tool.action
# in policy.yaml's approval_required_actions must pause BEFORE execution, not just
# the two hardcoded delete actions.


def _write_step(tool: str, action: str) -> PlanStep:
    return PlanStep(id="s1", tool=tool, action=action, arguments={})


@pytest.mark.parametrize(("tool", "action"), [
    ("google", "delete_event"),
    ("google", "delete_email"),
    ("github", "delete_repo"),
    ("github", "close_pr"),
    ("reddit", "post_comment"),
    ("reddit", "submit"),
])
def test_needs_approval_gates_every_policy_action(tool, action):
    assert _needs_approval([_write_step(tool, action)]) is True


def test_needs_approval_lets_reads_through():
    steps = [_write_step("google", "search_email"), _write_step("finnhub", "get_quote")]
    assert _needs_approval(steps) is False


def test_needs_approval_matches_on_tool_and_action_pair():
    # "submit" is gated only for reddit — the same action name on another tool
    # must not trip the gate (policy keys are tool.action pairs, not bare actions).
    assert _needs_approval([_write_step("finnhub", "submit")]) is False


def test_approval_draft_renders_every_gated_write():
    # The approval card must describe reddit/github writes too, not just calendar
    # deletes — and must not blow up now that the gate set comes from policy.yaml.
    steps = [_write_step("reddit", "submit"), _write_step("google", "delete_event")]
    draft = _approval_draft(_plan(steps))
    assert "reddit: submit" in draft.summary
    assert "delete" in draft.summary


# ── the planner-confidence gate (pre-execution, rule 6) ──────────────────────
# Policy default confidence_gate_threshold=medium: medium/low non-trusted-read
# steps pause; trusted reads never pause on confidence; policy actions always do.


def _conf_step(tool: str, action: str, confidence: str) -> PlanStep:
    return PlanStep(id="s1", tool=tool, action=action, arguments={},
                    confidence=confidence)  # type: ignore[arg-type]


def test_confidence_gate_pauses_medium_write():
    reasons = execute._steps_needing_approval(
        [_conf_step("google", "send_email", "medium")])
    assert reasons == {"s1": "low_confidence"}


def test_confidence_gate_lets_high_write_through():
    assert execute._steps_needing_approval(
        [_conf_step("google", "send_email", "high")]) == {}


def test_confidence_gate_never_pauses_trusted_reads():
    # finnhub.get_quote is in policy trusted_read_actions — low confidence is fine.
    assert execute._steps_needing_approval(
        [_conf_step("finnhub", "get_quote", "low")]) == {}


def test_policy_action_gates_even_at_high_confidence():
    reasons = execute._steps_needing_approval(
        [_conf_step("google", "delete_event", "high")])
    assert reasons == {"s1": "policy"}


def test_confidence_gate_threshold_low_ignores_medium(monkeypatch):
    monkeypatch.setattr(execute, "confidence_gate_threshold", lambda: "low")
    assert execute._steps_needing_approval(
        [_conf_step("google", "send_email", "medium")]) == {}
    assert execute._steps_needing_approval(
        [_conf_step("google", "send_email", "low")]) == {"s1": "low_confidence"}


def test_confidence_gate_threshold_off_only_policy_gates(monkeypatch):
    monkeypatch.setattr(execute, "confidence_gate_threshold", lambda: "off")
    assert execute._steps_needing_approval(
        [_conf_step("google", "send_email", "low")]) == {}
    assert execute._steps_needing_approval(
        [_conf_step("reddit", "submit", "high")]) == {"s1": "policy"}


def test_plan_step_confidence_defaults_high_for_legacy_state():
    # Old persisted state_json predates the field — it must rebuild as "high".
    step = PlanStep.model_validate({"id": "s1", "tool": "google", "action": "send_email"})
    assert step.confidence == "high"


def test_approval_draft_explains_low_confidence():
    draft = _approval_draft(_plan([_conf_step("google", "send_email", "medium")]))
    assert "Why am I being asked?" in draft.detail_markdown
    assert "medium confidence" in draft.detail_markdown
