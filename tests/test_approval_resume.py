"""HITL approval endpoint tests: reject, edit, approve-with-rebook, expiry.

``submit_approval`` is exercised directly with its module-level collaborators
monkeypatched — no HTTP server, DB, Redis, or compiled graph. The fake graph
echoes the state it was resumed with, so assertions inspect exactly what the
endpoint would hand to LangGraph.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import approvals
from app.api.approvals import ApprovalRequest
from app.core.state import AgentState, DraftResponse, ExecutionPlan, PlanStep, ToolResult


def _plan(step_id: str = "s1", **arguments) -> ExecutionPlan:
    step = PlanStep(id=step_id, tool="calendar", action="create_event", arguments=arguments)
    return ExecutionPlan(reasoning="r", steps=[step], strategy="sequential",
                         complexity_score=1, estimated_cost_usd=0.0, requires_approval=False)


def _draft(summary: str = "s") -> DraftResponse:
    return DraftResponse(summary=summary, detail_markdown="d",
                         actions_taken=[], actions_pending=[])


def _state(**overrides) -> AgentState:
    base: dict = dict(trace_id="t", user_id="u", session_id="s", user_request="do a thing")
    base.update(overrides)
    return AgentState(**base)


def _approval(hours: float = 1.0) -> dict:
    return {"token": "tok", "trace_id": "t", "decision": None,
            "expires_at": datetime.now(timezone.utc) + timedelta(hours=hours)}


@pytest.fixture
def wired(monkeypatch):
    """Wire submit_approval's collaborators to fakes; returns the recorder."""
    calls = SimpleNamespace(approval=_approval(), state=_state(),
                            saved=[], updated=[], invoked=[], published=[])

    async def fake_get_approval(token):
        return calls.approval

    async def fake_get_plan(trace_id):
        return {"trace_id": trace_id, "state_json": calls.state.model_dump_json()}

    async def fake_save_plan(state):
        calls.saved.append(state)

    async def fake_update(token, decision):
        calls.updated.append((token, decision))

    async def fake_publish(trace_id, state):
        calls.published.append(trace_id)

    class _FakeCompiled:
        async def ainvoke(self, state, config=None):
            calls.invoked.append(state)
            return state.model_dump()

    monkeypatch.setattr(approvals, "get_approval_by_token", fake_get_approval)
    monkeypatch.setattr(approvals, "get_plan_by_trace_id", fake_get_plan)
    monkeypatch.setattr(approvals, "save_plan", fake_save_plan)
    monkeypatch.setattr(approvals, "update_approval_status", fake_update)
    monkeypatch.setattr(approvals, "_publish_resume", fake_publish)
    monkeypatch.setattr(approvals, "compile_graph", lambda: _FakeCompiled())
    return calls


# ── token validation ─────────────────────────────────────────────────────────


async def test_unknown_token_is_404(wired):
    wired.approval = None
    with pytest.raises(HTTPException) as exc:
        await approvals.submit_approval("tok", ApprovalRequest(decision="approve"))
    assert exc.value.status_code == 404


async def test_expired_token_is_410(wired):
    wired.approval = _approval(hours=-1)
    with pytest.raises(HTTPException) as exc:
        await approvals.submit_approval("tok", ApprovalRequest(decision="approve"))
    assert exc.value.status_code == 410


async def test_missing_plan_row_is_404(wired, monkeypatch):
    async def no_plan(trace_id):
        return None

    monkeypatch.setattr(approvals, "get_plan_by_trace_id", no_plan)
    with pytest.raises(HTTPException) as exc:
        await approvals.submit_approval("tok", ApprovalRequest(decision="approve"))
    assert exc.value.status_code == 404


# ── decisions ────────────────────────────────────────────────────────────────


async def test_reject_persists_error_and_never_resumes_graph(wired):
    wired.state = _state(requires_approval=True, approval_token="tok")
    resp = await approvals.submit_approval("tok", ApprovalRequest(decision="reject"))
    assert resp.status == "rejected"
    assert wired.invoked == []  # graph must NOT run on reject
    saved = wired.saved[-1]
    assert saved.error == "rejected_by_user"
    assert saved.requires_approval is False
    assert wired.updated == [("tok", "rejected")]


async def test_edit_without_draft_is_400(wired):
    with pytest.raises(HTTPException) as exc:
        await approvals.submit_approval("tok", ApprovalRequest(decision="edit"))
    assert exc.value.status_code == 400


async def test_edit_replaces_draft_and_resumes(wired):
    wired.state = _state(draft=_draft("old"), requires_approval=True)
    resp = await approvals.submit_approval(
        "tok", ApprovalRequest(decision="edit", edited_draft=_draft("edited")))
    assert resp.status == "resumed"
    resumed = wired.invoked[0]
    assert resumed.draft is not None and resumed.draft.summary == "edited"
    assert resumed.requires_approval is False
    assert wired.updated == [("tok", "edit")]
    assert wired.published == ["t"]


async def test_approve_rebooks_calendar_conflict(wired):
    """Approving a clash rewrites the step to the suggested slot and re-executes."""
    conflict = ToolResult(
        step_id="s1", ok=True, latency_ms=1, error=None,
        output={"conflict": True,
                "requested": {"start": "2026-06-15T11:00:00+05:30",
                              "end": "2026-06-15T11:30:00+05:30"},
                "suggested": {"start": "2026-06-15T14:00:00+05:30",
                              "end": "2026-06-15T14:30:00+05:30"}},
    )
    wired.state = _state(
        plan=_plan("s1", start="2026-06-15T11:00:00+05:30", end="2026-06-15T11:30:00+05:30"),
        tool_results=[conflict], draft=_draft(), requires_approval=True, approval_token="tok",
    )
    resp = await approvals.submit_approval("tok", ApprovalRequest(decision="approve"))
    assert resp.status == "resumed"
    resumed = wired.invoked[0]
    step = resumed.plan.steps[0]
    assert step.arguments["start"] == "2026-06-15T14:00:00+05:30"
    assert step.arguments["end"] == "2026-06-15T14:30:00+05:30"
    # Cleared so the resumed graph re-enters the execute phase and books the slot.
    assert resumed.tool_results == []
    assert resumed.draft is None
    assert resumed.approval_token is None
    assert resumed.requires_approval is False


async def test_approve_without_conflict_resumes_unchanged(wired):
    wired.state = _state(plan=_plan("s1"), requires_approval=True, approval_token="tok")
    resp = await approvals.submit_approval("tok", ApprovalRequest(decision="approve"))
    assert resp.status == "resumed"
    resumed = wired.invoked[0]
    assert resumed.requires_approval is False
    assert resumed.approval_token == "tok"  # no rebook → token untouched


# ── _is_expired edge cases ───────────────────────────────────────────────────


def test_is_expired_none_never_expires():
    assert approvals._is_expired(None) is False


def test_is_expired_naive_datetime_is_treated_as_utc():
    assert approvals._is_expired(datetime(2000, 1, 1)) is True


def test_is_expired_parses_iso_strings():
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    assert approvals._is_expired(future) is False
    assert approvals._is_expired("not-a-date") is False  # unparseable → not expired
