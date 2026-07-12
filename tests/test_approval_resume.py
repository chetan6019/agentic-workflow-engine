"""HITL approval endpoint tests: reject, edit, approve, expiry, resumability.

``submit_approval`` is exercised directly with its module-level collaborators
monkeypatched — no HTTP server, DB, Redis, or compiled graph. Since the native
interrupt() migration, resume means ``ainvoke(Command(resume=payload))`` on the
paused thread; the fake graph records the Command so assertions inspect exactly
what would be handed to LangGraph. (The old rebuild-state-and-reinvoke tests —
reject-without-graph, API-side calendar rebook — died with that mechanism; the
rebook now lives in the execute node and is covered by test_interrupt_flow.py.)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from langgraph.types import Command

from app.api import approvals
from app.api.approvals import ApprovalRequest
from app.core.state import AgentState, DraftResponse


def _draft(summary: str = "s") -> DraftResponse:
    return DraftResponse(summary=summary, detail_markdown="d",
                         actions_taken=[], actions_pending=[])


def _approval(hours: float = 1.0) -> dict:
    return {"token": "tok", "trace_id": "t", "decision": None,
            "expires_at": datetime.now(timezone.utc) + timedelta(hours=hours)}


# A request with no authenticated caller — ownership enforcement is skipped, so
# these endpoint-level tests exercise the decision logic, not the cross-user gate.
_REQ = SimpleNamespace(state=SimpleNamespace(user_id=None))


@pytest.fixture
def wired(monkeypatch):
    """Wire submit_approval's collaborators to fakes; returns the recorder."""
    calls = SimpleNamespace(approval=_approval(), saved=[], updated=[],
                            resumed=[], discarded=[], has_pending_node=True)

    async def fake_get_approval(token):
        return calls.approval

    async def fake_get_plan(trace_id):
        return {"trace_id": trace_id, "user_id": "u", "state_json": "{}"}

    async def fake_save_plan(state):
        calls.saved.append(state)

    async def fake_update(token, decision):
        calls.updated.append((token, decision))

    async def fake_discard(trace_id):
        calls.discarded.append(trace_id)

    class _FakeCompiled:
        async def aget_state(self, config):
            # A resumable thread has a pending (interrupted) node in `next`.
            nxt = ("execute",) if calls.has_pending_node else ()
            return SimpleNamespace(next=nxt)

        async def ainvoke(self, cmd, config=None):
            calls.resumed.append((cmd, config))
            decision = (cmd.resume or {}).get("decision")
            state = AgentState(trace_id=config["configurable"]["thread_id"],
                               user_id="u", session_id="s", user_request="do a thing")
            if decision == "reject":
                state.error = "rejected_by_user"
            else:
                state.draft = _draft("resumed result")
            return state.model_dump()

    monkeypatch.setattr(approvals, "get_approval_by_token", fake_get_approval)
    monkeypatch.setattr(approvals, "get_plan_by_trace_id", fake_get_plan)
    monkeypatch.setattr(approvals, "save_plan", fake_save_plan)
    monkeypatch.setattr(approvals, "update_approval_status", fake_update)
    monkeypatch.setattr(approvals, "discard_thread", fake_discard)
    monkeypatch.setattr(approvals, "compile_graph", lambda: _FakeCompiled())
    return calls


# ── token validation ─────────────────────────────────────────────────────────


async def test_unknown_token_is_404(wired):
    wired.approval = None
    with pytest.raises(HTTPException) as exc:
        await approvals.submit_approval("tok", ApprovalRequest(decision="approve"), _REQ)
    assert exc.value.status_code == 404


async def test_expired_token_is_410(wired):
    wired.approval = _approval(hours=-1)
    with pytest.raises(HTTPException) as exc:
        await approvals.submit_approval("tok", ApprovalRequest(decision="approve"), _REQ)
    assert exc.value.status_code == 410


async def test_missing_plan_row_is_404(wired, monkeypatch):
    async def no_plan(trace_id):
        return None

    monkeypatch.setattr(approvals, "get_plan_by_trace_id", no_plan)
    with pytest.raises(HTTPException) as exc:
        await approvals.submit_approval("tok", ApprovalRequest(decision="approve"), _REQ)
    assert exc.value.status_code == 404


async def test_cross_user_caller_is_403(wired, monkeypatch):
    async def other_owner(trace_id):
        return {"trace_id": trace_id, "user_id": "owner-2", "state_json": "{}"}

    monkeypatch.setattr(approvals, "get_plan_by_trace_id", other_owner)
    req = SimpleNamespace(state=SimpleNamespace(user_id="user-1"))
    with pytest.raises(HTTPException) as exc:
        await approvals.submit_approval("tok", ApprovalRequest(decision="approve"), req)
    assert exc.value.status_code == 403


# ── resumability ─────────────────────────────────────────────────────────────


async def test_missing_checkpoint_is_409(wired):
    # Restart on the memory backend / pre-migration approval: nothing to resume.
    wired.has_pending_node = False
    with pytest.raises(HTTPException) as exc:
        await approvals.submit_approval("tok", ApprovalRequest(decision="approve"), _REQ)
    assert exc.value.status_code == 409
    assert exc.value.detail == "approval_not_resumable"
    assert wired.resumed == []


# ── decisions (all three flow through Command(resume=...)) ───────────────────


async def test_approve_resumes_with_command(wired):
    resp = await approvals.submit_approval("tok", ApprovalRequest(decision="approve"), _REQ)
    assert resp.status == "resumed"
    cmd, config = wired.resumed[0]
    assert isinstance(cmd, Command)
    assert cmd.resume == {"decision": "approve", "edited_draft": None}
    assert config["configurable"]["thread_id"] == "t"
    assert wired.updated == [("tok", "approve")]
    assert wired.discarded == ["t"]  # resumed run is terminal
    assert wired.saved                # final state persisted


async def test_reject_flows_through_the_graph(wired):
    resp = await approvals.submit_approval("tok", ApprovalRequest(decision="reject"), _REQ)
    assert resp.status == "rejected"
    cmd, _ = wired.resumed[0]  # one mechanism for all three decisions
    assert cmd.resume["decision"] == "reject"
    assert wired.saved[-1].error == "rejected_by_user"
    assert wired.updated == [("tok", "rejected")]  # DB records past-tense
    assert wired.discarded == ["t"]


async def test_edit_without_draft_is_400(wired):
    with pytest.raises(HTTPException) as exc:
        await approvals.submit_approval("tok", ApprovalRequest(decision="edit"), _REQ)
    assert exc.value.status_code == 400
    assert wired.resumed == []  # rejected before touching the graph


async def test_edit_carries_the_edited_draft(wired):
    resp = await approvals.submit_approval(
        "tok", ApprovalRequest(decision="edit", edited_draft=_draft("edited")), _REQ)
    assert resp.status == "resumed"
    cmd, _ = wired.resumed[0]
    assert cmd.resume["edited_draft"]["summary"] == "edited"
    assert wired.updated == [("tok", "edit")]


# ── _is_expired edge cases ───────────────────────────────────────────────────


def test_is_expired_none_never_expires():
    assert approvals._is_expired(None) is False


def test_is_expired_naive_datetime_is_treated_as_utc():
    assert approvals._is_expired(datetime(2000, 1, 1)) is True


def test_is_expired_parses_iso_strings():
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    assert approvals._is_expired(future) is False
    assert approvals._is_expired("not-a-date") is False  # unparseable → not expired