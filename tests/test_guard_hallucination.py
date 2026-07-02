"""Guard hallucination-mitigation tests (REVIEW.md R31).

guardrails_node is driven with the judge LLM, compose, and get_token faked, so
the re-compose-then-warn path is exercised with no LLM, DB, or Qdrant. The fake
judge pops verdicts from a shared queue so the initial judge and the post-
recompose re-judge can return different scores.
"""

from __future__ import annotations

import pytest

from app.core.state import (
    AgentState,
    DraftResponse,
    ExecutionPlan,
    GuardVerdict,
    PlanStep,
    ToolResult,
)
from app.orchestration.nodes import guard
from app.prompts import build_composer_messages, build_guard_judge_messages


def _verdict(risk: float) -> GuardVerdict:
    return GuardVerdict(tone_fit=0.9, hallucination_risk=risk, instruction_adherence=0.9)


def _state(detail: str = "original body") -> AgentState:
    plan = ExecutionPlan(
        reasoning="r", steps=[PlanStep(id="s1", tool="gmail", action="search_email")],
        strategy="sequential", complexity_score=1, estimated_cost_usd=0.0,
        requires_approval=False)
    return AgentState(
        trace_id="t", user_id="u", session_id="s", user_request="summarise my inbox",
        plan=plan,
        tool_results=[ToolResult(step_id="s1", ok=True, output={"messages": []},
                                 error=None, latency_ms=3)],
        draft=DraftResponse(summary="orig", detail_markdown=detail,
                            actions_taken=[], actions_pending=[]),
    )


class _FakeJudge:
    """Returns queued verdicts in order; shared across get_structured_llm calls."""

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)

    async def ainvoke(self, _messages):
        return self._verdicts.pop(0)


@pytest.fixture
def wire(monkeypatch):
    """Fake get_token + a recompose stub; returns a configure() for the judge queue."""
    async def fake_token(_user_id, _tool):
        return "tok"

    monkeypatch.setattr(guard, "get_token", fake_token)

    recomposed = DraftResponse(summary="re", detail_markdown="recomposed body",
                               actions_taken=[], actions_pending=[])

    async def fake_compose(_state, judge_feedback=None):
        fake_compose.feedback = judge_feedback
        return recomposed

    fake_compose.feedback = None
    monkeypatch.setattr(guard, "compose", fake_compose)

    def configure(verdicts):
        judge = _FakeJudge(verdicts)
        monkeypatch.setattr(guard, "get_structured_llm", lambda *a, **k: judge)
        return fake_compose

    return configure


async def test_low_risk_leaves_draft_untouched(wire):
    wire([_verdict(0.1)])
    state = await guard.guardrails_node(_state())
    assert state.draft.detail_markdown == "original body"
    assert state.degraded == []
    assert state.confidence > 0


async def test_mid_band_risk_no_longer_mitigates(wire):
    """Regression: risk 0.8 used to trip mitigation (limit 0.7); the limit is now 0.85, so a
    faithfully-summarised draft scored 0.8 finalizes untouched — no re-compose, no warning."""
    wire([_verdict(0.8)])
    state = await guard.guardrails_node(_state())
    assert state.draft.detail_markdown == "original body"
    assert state.degraded == []


async def test_high_then_low_recomposes_without_warning(wire):
    """Re-composition resolves the risk → new draft, no warning, no degraded flag."""
    compose_stub = wire([_verdict(0.9), _verdict(0.1)])
    state = await guard.guardrails_node(_state())
    assert state.draft.detail_markdown == "recomposed body"  # replaced
    assert "⚠️" not in state.draft.detail_markdown
    assert state.degraded == []
    assert "unsupported claims" in (compose_stub.feedback or "")  # feedback threaded


async def test_high_then_high_warns_and_flags_degraded(wire):
    """Re-composition fails to resolve → warning prepended + degraded flag."""
    wire([_verdict(0.9), _verdict(0.95)])
    state = await guard.guardrails_node(_state())
    assert state.draft.detail_markdown.startswith("> ⚠️")
    assert "recomposed body" in state.draft.detail_markdown
    assert state.degraded == ["unverified_hallucination_risk"]


async def test_recompose_transport_failure_keeps_original(wire, monkeypatch):
    """A failing re-composition keeps the original draft and records the degradation."""
    wire([_verdict(0.9)])  # only the initial judge runs; re-judge never reached

    async def boom(_state, judge_feedback=None):
        raise RuntimeError("llm down")

    monkeypatch.setattr(guard, "compose", boom)
    state = await guard.guardrails_node(_state("keep me"))
    assert state.draft.detail_markdown == "keep me"
    assert state.degraded == ["hallucination_recompose_failed"]


# ── prompt wiring ────────────────────────────────────────────────────────────


def test_judge_prompt_includes_evidence():
    draft = DraftResponse(summary="s", detail_markdown="d", actions_taken=[], actions_pending=[])
    results = [ToolResult(step_id="s1", ok=True, output={"count": 2}, error=None, latency_ms=1)]
    msgs = build_guard_judge_messages(draft=draft, user_request="q", tool_results=results)
    human = msgs[-1].content
    assert "<evidence>" in human and "count" in human


def test_composer_revision_note_only_when_feedback_given():
    base = dict(plan={"steps": []}, tool_results=[], preferences=[], user_request="q",
                tz_name="Asia/Kolkata")
    assert "<revision_note>" not in build_composer_messages(**base)[-1].content
    with_fb = build_composer_messages(**base, judge_feedback="ground your claims")[-1].content
    assert "<revision_note>" in with_fb and "ground your claims" in with_fb
