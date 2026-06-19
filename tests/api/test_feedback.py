"""Async route tests for POST /v1/feedback/{trace_id}."""

from __future__ import annotations

import app.api.feedback as feedback_mod


async def test_feedback_persisted_for_own_run(client, backend, auth_headers):
    backend.add_plan("t1", "user-1", "{}")
    resp = await client.post("/v1/feedback/t1", json={"score": 5, "comment": "great"},
                             headers=auth_headers("user-1"))
    assert resp.status_code == 200
    assert backend.feedback == [{"trace_id": "t1", "score": 5, "comment": "great"}]


async def test_thumbs_down_without_comment(client, backend, auth_headers):
    backend.add_plan("t1", "user-1", "{}")
    resp = await client.post("/v1/feedback/t1", json={"score": 1},
                             headers=auth_headers("user-1"))
    assert resp.status_code == 200
    assert backend.feedback[0]["score"] == 1


async def test_missing_jwt_is_401(client, backend):
    assert (await client.post("/v1/feedback/t1", json={"score": 5})).status_code == 401


async def test_feedback_on_unknown_run_is_404(client, backend, auth_headers):
    resp = await client.post("/v1/feedback/ghost", json={"score": 5},
                             headers=auth_headers("user-1"))
    assert resp.status_code == 404


async def test_feedback_on_other_users_run_is_403(client, backend, auth_headers):
    backend.add_plan("t1", "owner-2", "{}")
    resp = await client.post("/v1/feedback/t1", json={"score": 5},
                             headers=auth_headers("user-1"))
    assert resp.status_code == 403


async def test_feedback_rate_limited(client, backend, auth_headers, monkeypatch):
    backend.add_plan("t1", "user-1", "{}")

    async def deny(identity, limit, window_sec=60):
        return False

    monkeypatch.setattr(feedback_mod, "check_rate_limit", deny)
    resp = await client.post("/v1/feedback/t1", json={"score": 5},
                             headers=auth_headers("user-1"))
    assert resp.status_code == 429
