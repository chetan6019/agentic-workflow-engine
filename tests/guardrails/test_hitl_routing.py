"""Integration: an OUTPUT guardrail flag routes the run through the HITL path.

guardrails_node is driven with the judge, get_token, create_approval and save_plan
faked — no LLM, DB, Redis, or Qdrant. A draft citing a doc_id absent from the
retrieved context trips ``hallucination_citation`` (flag_for_approval), and the
node must pause for approval exactly like the existing PII/clash gates do.
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

    async def fake_create_approval(_trace_id):
        return "appr-tok"

    async def fake_save_plan(state):
        saved.append(state)

    class _Judge:
        async def ainvoke(self, _messages):
            return GuardVerdict(tone_fit=0.9, hallucination_risk=0.1, instruction_adherence=0.9)

    monkeypatch.setattr(guard, "get_token", fake_token)
    monkeypatch.setattr(guard, "create_approval", fake_create_approval)
    monkeypatch.setattr(guard, "save_plan", fake_save_plan)
    monkeypatch.setattr(guard, "get_structured_llm", lambda *a, **k: _Judge())
    return saved


async def test_uncited_draft_pauses_for_approval(wired):
    state = await guard.guardrails_node(_state(citations=["doc-ghost"]))
    assert state.requires_approval is True
    assert state.approval_token == "appr-tok"
    assert state.error is None
    assert state.verdict is None  # router falls through to END (paused)
    assert wired  # save_plan persisted the paused state


async def test_clean_draft_does_not_pause(wired):
    state = await guard.guardrails_node(_state(citations=[]))
    assert state.requires_approval is False
    assert state.approval_token is None
    assert state.verdict == "finalize"
