"""End-to-end interrupt()/Command(resume) flow on a REAL compiled graph.

The graph is compiled fresh on a MemorySaver with each node's collaborators
faked (no LLM, DB, Redis, MCP); only the planner node is replaced by a
passthrough, since the seeded state already carries the plan under test.
These tests pin the properties the native-interrupt migration must guarantee:

- a gated step pauses BEFORE its tool runs;
- resume replays the node without duplicating side effects (one approval token,
  tools execute exactly once);
- reject resumes through the same mechanism and ends the run;
- an approved calendar-conflict rebook re-executes with the suggested slot.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.core.state import (
    AgentState,
    DraftResponse,
    ExecutionPlan,
    GuardVerdict,
    PlanStep,
    ToolResult,
)
from app.orchestration import graph
from app.orchestration.nodes import compose as node_compose
from app.orchestration.nodes import execute, guard
from app.orchestration.nodes import intake as node_intake
from app.orchestration.nodes import respond as node_respond
from app.orchestration.nodes import retrieve as node_retrieve


def _plan(tool: str, action: str, confidence: str = "high", **arguments: Any) -> ExecutionPlan:
    step = PlanStep(id="s1", tool=tool, action=action, arguments=arguments,
                    confidence=confidence)  # type: ignore[arg-type]
    return ExecutionPlan(reasoning="r", steps=[step], strategy="sequential",
                         complexity_score=1, estimated_cost_usd=0.0, requires_approval=False)


def _state(plan: ExecutionPlan) -> AgentState:
    return AgentState(trace_id="t-int", user_id="u", session_id="s",
                      user_request="do the thing", plan=plan)


class FakeMCP:
    """Records tool calls; a conflict_until_start arg simulates a calendar clash."""

    def __init__(self, conflict_until_start: str | None = None,
                 suggested: dict | None = None) -> None:
        self.calls: list[dict] = []
        self.conflict_until_start = conflict_until_start
        self.suggested = suggested

    async def call_tool(self, server: str, tool: str, args: dict,
                        trace_id: str, step_id: str) -> ToolResult:
        self.calls.append(dict(args))
        if (self.conflict_until_start is not None
                and args.get("start") == self.conflict_until_start):
            return ToolResult(step_id=step_id, ok=True, error=None, latency_ms=1,
                              output={"conflict": True,
                                      "requested": {"start": args.get("start")},
                                      "conflicting": [{"summary": "busy"}],
                                      "suggested": self.suggested})
        return ToolResult(step_id=step_id, ok=True, error=None, latency_ms=1,
                          output={"ok": True})


@pytest.fixture
def wired(monkeypatch):
    """Fake every external collaborator; return a recorder namespace."""
    from types import SimpleNamespace

    rec = SimpleNamespace(mcp=FakeMCP(), saved=[], tokens_minted=0)

    async def fake_save_plan(state):
        rec.saved.append(state)

    async def fake_save_tool_call(trace_id, result):
        return None

    async def fake_get_or_create(trace_id):
        # get-or-create semantics: mint once, then keep returning the same token.
        if rec.tokens_minted == 0:
            rec.tokens_minted = 1
        return "tok-1"

    async def fake_compose(state, judge_feedback=None):
        return DraftResponse(summary="composed", detail_markdown="done",
                             actions_taken=["did it"], actions_pending=[], citations=[])

    async def fake_get_token(_user_id, _tool):
        return "integration-token"

    class _Judge:
        async def ainvoke(self, _messages):
            return GuardVerdict(tone_fit=0.9, hallucination_risk=0.1,
                                instruction_adherence=0.9)

    async def fake_evaluate_output(text, **kwargs):
        return SimpleNamespace(text=text, allowed=True, requires_approval=False,
                               blocked_rules=[], hits=[])

    async def fake_get_session(session_id):
        return {"session_id": session_id}

    async def fake_retrieve(state):
        return []

    async def passthrough_planner_node(state):
        # The seeded state already carries the plan under test — no LLM needed.
        return state

    monkeypatch.setattr(node_intake, "get_session", fake_get_session)
    monkeypatch.setattr(node_retrieve, "retrieve", fake_retrieve)
    monkeypatch.setattr(graph, "planner_node", passthrough_planner_node)
    monkeypatch.setattr(execute, "save_plan", fake_save_plan)
    monkeypatch.setattr(execute, "save_tool_call", fake_save_tool_call)
    monkeypatch.setattr(execute, "get_or_create_approval", fake_get_or_create)
    monkeypatch.setattr(execute, "get_mcp_client", lambda: rec.mcp)
    monkeypatch.setattr(node_compose, "compose", fake_compose)
    monkeypatch.setattr(node_compose, "save_plan", fake_save_plan)
    monkeypatch.setattr(node_respond, "save_plan", fake_save_plan)
    monkeypatch.setattr(node_respond, "spawn", lambda coro: coro.close())
    monkeypatch.setattr(guard, "save_plan", fake_save_plan)
    monkeypatch.setattr(guard, "get_token", fake_get_token)
    monkeypatch.setattr(guard, "get_structured_llm", lambda *a, **k: _Judge())
    monkeypatch.setattr(guard, "evaluate_output", fake_evaluate_output)
    return rec


def _compiled():
    """Fresh graph on a fresh MemorySaver (not the lru-cached singleton)."""
    return graph._build_graph().compile(checkpointer=MemorySaver())


_CFG = {"configurable": {"thread_id": "t-int"}}


async def test_gated_step_pauses_before_tool_runs(wired):
    compiled = _compiled()
    result = await compiled.ainvoke(_state(_plan("reddit", "submit")), config=_CFG)
    assert "__interrupt__" in result          # run paused via native interrupt()
    assert wired.mcp.calls == []              # the write did NOT execute
    assert wired.saved                        # paused state persisted for the UI
    assert wired.saved[-1].approval_token == "tok-1"


async def test_approve_resume_executes_exactly_once(wired):
    compiled = _compiled()
    await compiled.ainvoke(_state(_plan("reddit", "submit")), config=_CFG)
    final = await compiled.ainvoke(
        Command(resume={"decision": "approve", "edited_draft": None}), config=_CFG)
    state = AgentState.model_validate(final)
    assert state.error is None
    assert state.draft is not None and state.draft.summary == "composed"
    assert len(wired.mcp.calls) == 1          # replay-safe: the write ran ONCE
    assert wired.tokens_minted == 1           # and only one token was ever minted


async def test_reject_resume_ends_run_without_executing(wired):
    compiled = _compiled()
    await compiled.ainvoke(_state(_plan("reddit", "submit")), config=_CFG)
    final = await compiled.ainvoke(
        Command(resume={"decision": "reject", "edited_draft": None}), config=_CFG)
    state = AgentState.model_validate(final)
    assert state.error == "rejected_by_user"
    assert wired.mcp.calls == []              # rejected write never executed


async def test_medium_confidence_write_pauses(wired):
    # The confidence gate: send_email is not a policy action, but medium confidence
    # on a non-trusted-read step must pause pre-execution (rule 6).
    compiled = _compiled()
    result = await compiled.ainvoke(
        _state(_plan("google", "send_email", confidence="medium")), config=_CFG)
    assert "__interrupt__" in result
    assert wired.mcp.calls == []


async def test_calendar_conflict_rebook_re_executes_with_suggested_slot(wired):
    original = "2026-07-10T11:00:00+05:30"
    suggested = {"start": "2026-07-10T14:00:00+05:30", "end": "2026-07-10T14:30:00+05:30"}
    wired.mcp = FakeMCP(conflict_until_start=original, suggested=suggested)
    compiled = _compiled()
    result = await compiled.ainvoke(
        _state(_plan("google", "create_event", start=original, end="x")), config=_CFG)
    assert "__interrupt__" in result          # paused on the clash
    final = await compiled.ainvoke(
        Command(resume={"decision": "approve", "edited_draft": None}), config=_CFG)
    state = AgentState.model_validate(final)
    assert state.error is None
    assert state.plan is not None
    assert state.plan.steps[0].arguments["start"] == suggested["start"]
    # Last executed call used the suggested slot (earlier calls hit the clash).
    assert wired.mcp.calls[-1]["start"] == suggested["start"]
    assert state.draft is not None and state.draft.summary == "composed"