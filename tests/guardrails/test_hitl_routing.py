"""Integration: an OUTPUT guardrail flag routes the run through the HITL path.

guardrails_node is driven with the judge, get_token, get_or_create_approval,
save_plan — and, since the native-interrupt migration, ``interrupt`` itself —
faked. A draft citing a doc_id absent from the retrieved context trips
``hallucination_citation`` (flag_for_approval); the node must pause via
interrupt() and, on resume, honor the human decision.
"""

from __future__ import annotations

import pytest

from app.core.state import (
    AgentState,
    DraftResponse,
    ExecutionPlan,
    GuardVerdict,
    PlanStep,
)
from app.orchestration.nodes import guard


class FakeInterrupt(Exception):
    """Stands in for LangGraph's GraphInterrupt on the first (pause) pass."""


def _state(citations: list[str]) -> AgentState:
    plan = ExecutionPlan(
        reasoning="r", steps=[PlanStep(id="s1", tool="gmail", action="search_email")],
        strategy="sequential", complexity_score=1, estimated_cost_usd=0.0,
        requires_approval=False)
    return AgentState(
        trace_id="t", user_id="u", session_id="s", user_request="summarise inbox",
        plan=plan,
        draft=DraftResponse(summary="s", detail_markdown="body", actions_taken=[],
                            actions_pending=[], citations=citations),
    )


@pytest.fixture
def wired(monkeypatch):
    saved: list[AgentState] = []

    async def fake_token(_user_id, _tool):
        return "tok"

    async def fake_get_or_create(_trace_id):
        return "appr-tok"

    async def fake_save_plan(state):
        saved.append(state)

    class _Judge:
        async def ainvoke(self, _messages):
            return GuardVerdict(tone_fit=0.9, hallucination_risk=0.1, instruction_adherence=0.9)

    monkeypatch.setattr(guard, "get_token", fake_token)
    monkeypatch.setattr(guard, "get_or_create_approval", fake_get_or_create)
    monkeypatch.setattr(guard, "save_plan", fake_save_plan)
    monkeypatch.setattr(guard, "get_structured_llm", lambda *a, **k: _Judge())
    return saved


async def test_uncited_draft_pauses_via_interrupt(wired, monkeypatch):
    # First pass: interrupt() raises (like GraphInterrupt) — the node must have
    # persisted the token + paused state BEFORE the raise so the UI can render it.
    def raising_interrupt(payload):
        assert payload["kind"] == "output_approval"
        assert payload["token"] == "appr-tok"
        raise FakeInterrupt()

    monkeypatch.setattr(guard, "interrupt", raising_interrupt)
    with pytest.raises(FakeInterrupt):
        await guard.guardrails_node(_state(citations=["doc-ghost"]))
    assert wired  # save_plan persisted the paused state
    paused = wired[-1]
    assert paused.requires_approval is True
    assert paused.approval_token == "appr-tok"


async def test_uncited_draft_approve_resume_finalizes(wired, monkeypatch):
    # Resume replay: interrupt() RETURNS the decision payload and the node continues.
    monkeypatch.setattr(guard, "interrupt", lambda payload: {"decision": "approve"})
    state = await guard.guardrails_node(_state(citations=["doc-ghost"]))
    assert state.error is None
    assert state.requires_approval is False
    assert state.verdict == "finalize"


async def test_uncited_draft_reject_resume_sets_error(wired, monkeypatch):
    monkeypatch.setattr(guard, "interrupt", lambda payload: {"decision": "reject"})
    state = await guard.guardrails_node(_state(citations=["doc-ghost"]))
    assert state.error == "rejected_by_user"
    assert state.requires_approval is False


async def test_clean_draft_does_not_pause(wired, monkeypatch):
    def never_interrupt(payload):  # a clean draft must not reach interrupt at all
        raise AssertionError("interrupt() called for a clean draft")

    monkeypatch.setattr(guard, "interrupt", never_interrupt)
    state = await guard.guardrails_node(_state(citations=[]))
    assert state.requires_approval is False
    assert state.approval_token is None
    assert state.verdict == "finalize"