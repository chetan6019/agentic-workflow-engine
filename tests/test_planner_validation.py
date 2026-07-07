"""Planner plan-validation tests (REVIEW.md R32).

planner_node is driven with fetch_tool_specs, call_planner_llm, and get_settings
faked, so the validate → correct-once → fail-honestly path runs with no LLM,
Qdrant, or settings file.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.state import AgentState, ExecutionPlan, PlanStep, ToolSpec
from app.orchestration.nodes import planner


def _spec(name: str, server: str) -> ToolSpec:
    return ToolSpec(name=name, description="d", server=server)


def _plan(tool: str, action: str) -> ExecutionPlan:
    return ExecutionPlan(
        reasoning="r", steps=[PlanStep(id="s1", tool=tool, action=action)],
        strategy="sequential", complexity_score=1, estimated_cost_usd=0.0,
        requires_approval=False)


def _state() -> AgentState:
    return AgentState(trace_id="t", user_id="u", session_id="s",
                      user_request="send an email")


@pytest.fixture
def wire(monkeypatch):
    """Fake settings + tool catalog; returns a configure(specs, plans) helper."""
    monkeypatch.setattr(planner, "get_settings",
                        lambda: SimpleNamespace(default_tz="Asia/Kolkata"))

    def configure(specs, plans):
        async def fake_specs(_query, _k):
            return specs

        invoked = []

        async def fake_invoke(_role, messages, _meta):
            invoked.append(messages)
            return plans.pop(0)

        monkeypatch.setattr(planner, "fetch_tool_specs", fake_specs)
        monkeypatch.setattr(planner, "call_planner_llm", fake_invoke)
        return invoked

    return configure


# ── find_invalid_plan_steps (pure) ───────────────────────────────────────────


def test_invalid_steps_accepts_known_pair():
    assert planner.find_invalid_plan_steps(_plan("gmail", "send_email"),
                                           [_spec("send_email", "gmail")]) == []


def test_invalid_steps_accepts_swapped_pair():
    # MCP client resolves swapped server/action, so validation tolerates it.
    assert planner.find_invalid_plan_steps(_plan("send_email", "gmail"),
                                           [_spec("send_email", "gmail")]) == []


def test_invalid_steps_flags_unknown_action():
    bad = planner.find_invalid_plan_steps(_plan("gmail", "teleport"), [_spec("send_email", "gmail")])
    assert bad == ["s1:gmail/teleport"]


# ── planner_node ─────────────────────────────────────────────────────────────


async def test_valid_plan_passes_without_correction(wire):
    invoked = wire([_spec("send_email", "gmail")], [_plan("gmail", "send_email")])
    state = await planner.planner_node(_state())
    assert state.error is None
    assert state.plan is not None
    assert len(invoked) == 1  # no corrective re-plan


async def test_invalid_plan_is_corrected_on_retry(wire):
    invoked = wire([_spec("send_email", "gmail")],
                   [_plan("gmail", "bogus"), _plan("gmail", "send_email")])
    state = await planner.planner_node(_state())
    assert state.error is None
    assert state.plan.steps[0].action == "send_email"
    assert len(invoked) == 2
    assert "s1:gmail/bogus" in invoked[1][-1].content  # feedback names the bad step


async def test_still_invalid_after_retry_sets_error(wire):
    wire([_spec("send_email", "gmail")],
         [_plan("gmail", "bogus"), _plan("gmail", "stillbogus")])
    state = await planner.planner_node(_state())
    assert state.error is not None
    assert state.error.startswith("plan_validation_failed")
    assert state.plan is None


async def test_empty_catalog_skips_validation(wire):
    invoked = wire([], [_plan("gmail", "anything")])
    state = await planner.planner_node(_state())
    assert state.error is None
    assert state.plan is not None
    assert len(invoked) == 1  # no validation attempted


# ── no-op / greeting plans (regression: complexity_score 0 must validate) ─────


def test_noop_plan_with_zero_complexity_is_valid():
    # Regression: a greeting / standalone question yields steps=[] and complexity_score=0.
    # The schema previously required complexity_score >= 1, so the whole run errored on a
    # plain "hi". An empty no-tool plan must validate cleanly.
    plan = ExecutionPlan(reasoning="greeting; no tools", steps=[], strategy="sequential",
                         complexity_score=0, estimated_cost_usd=0.0, requires_approval=False)
    assert plan.steps == [] and plan.complexity_score == 0


async def test_noop_plan_passes_for_greeting(wire):
    noop = ExecutionPlan(reasoning="greeting", steps=[], strategy="sequential",
                         complexity_score=0, estimated_cost_usd=0.0, requires_approval=False)
    invoked = wire([_spec("send_email", "google")], [noop])
    state = await planner.planner_node(_state())
    assert state.error is None
    assert state.plan is not None and state.plan.steps == []
    assert len(invoked) == 1  # empty steps → nothing to validate, no correction
