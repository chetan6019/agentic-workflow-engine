"""Async route tests for GET /v1/approvals and POST /v1/approvals/{token}."""

from __future__ import annotations

from app.core.state import AgentState


def _state_json(user_id: str, trace_id: str = "t") -> str:
    return AgentState(trace_id=trace_id, user_id=user_id, session_id="s",
                      user_request="do a thing").model_dump_json()


async def test_list_pending_is_owner_scoped(client, backend, auth_headers):
    backend.add_plan("t1", "user-1", _state_json("user-1", "t1"))
    backend.add_plan("t2", "user-2", _state_json("user-2", "t2"))
    backend.add_approval("tok1", "t1")
    backend.add_approval("tok2", "t2")
    resp = await client.get("/v1/approvals?status=pending", headers=auth_headers("user-1"))
    assert resp.status_code == 200
    tokens = [a["token"] for a in resp.json()["approvals"]]
    assert tokens == ["tok1"]


async def test_list_requires_auth(client, backend):
    assert (await client.get("/v1/approvals")).status_code == 401


async def test_expired_pending_auto_rejects_on_read(client, backend, auth_headers):
    backend.add_plan("t1", "user-1", _state_json("user-1", "t1"))
    backend.add_approval("tok1", "t1", hours=-1)  # already expired
    resp = await client.get("/v1/approvals", headers=auth_headers("user-1"))
    assert resp.json()["approvals"] == []
    assert backend.approvals["tok1"]["decision"] == "rejected"  # auto-rejected


async def test_approve_resumes_graph(client, backend, auth_headers):
    backend.add_plan("t1", "user-1", _state_json("user-1", "t1"))
    backend.add_approval("tok1", "t1")
    resp = await client.post("/v1/approvals/tok1", json={"decision": "approve"},
                             headers=auth_headers("user-1"))
    assert resp.status_code == 200 and resp.json()["status"] == "resumed"
    assert backend.approvals["tok1"]["decision"] == "approve"


async def test_reject_closes_run_without_resume(client, backend, auth_headers):
    backend.add_plan("t1", "user-1", _state_json("user-1", "t1"))
    backend.add_approval("tok1", "t1")
    resp = await client.post("/v1/approvals/tok1", json={"decision": "reject"},
                             headers=auth_headers("user-1"))
    assert resp.status_code == 200 and resp.json()["status"] == "rejected"
    assert backend.approvals["tok1"]["decision"] == "rejected"


async def test_expired_token_post_is_410(client, backend, auth_headers):
    backend.add_plan("t1", "user-1", _state_json("user-1", "t1"))
    backend.add_approval("tok1", "t1", hours=-1)
    resp = await client.post("/v1/approvals/tok1", json={"decision": "approve"},
                             headers=auth_headers("user-1"))
    assert resp.status_code == 410


async def test_cross_user_approve_is_403(client, backend, auth_headers):
    backend.add_plan("t1", "user-1", _state_json("user-1", "t1"))
    backend.add_approval("tok1", "t1")
    resp = await client.post("/v1/approvals/tok1", json={"decision": "approve"},
                             headers=auth_headers("intruder"))
    assert resp.status_code == 403
