"""Async route tests for POST /v1/feedback/{trace_id}."""

from __future__ import annotations

from types import SimpleNamespace

import app.api.feedback as feedback_mod


class _FakeLangfuse:
    """Captures create_score calls; optionally raises to test best-effort."""

    def __init__(self, raise_on_score: bool = False) -> None:
        self.raise_on_score = raise_on_score
        self.scores: list[dict] = []

    def create_score(self, **kwargs) -> None:
        if self.raise_on_score:
            raise RuntimeError("langfuse down")
        self.scores.append(kwargs)


def _enable_langfuse(monkeypatch, fake: _FakeLangfuse) -> None:
    monkeypatch.setattr(feedback_mod, "get_settings", lambda: SimpleNamespace(
        langfuse_public_key="pk", langfuse_secret_key="sk"))
    monkeypatch.setattr(feedback_mod, "_langfuse_client", lambda: fake)


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


async def test_feedback_mirrored_to_langfuse(client, backend, auth_headers, monkeypatch):
    backend.add_plan("t1", "user-1", "{}")
    fake = _FakeLangfuse()
    _enable_langfuse(monkeypatch, fake)
    resp = await client.post("/v1/feedback/t1", json={"score": 2, "comment": "meh"},
                             headers=auth_headers("user-1"))
    assert resp.status_code == 200
    assert fake.scores == [{"name": "user_feedback", "value": 2.0,
                            "trace_id": "t1", "comment": "meh"}]


async def test_langfuse_failure_does_not_fail_request(client, backend, auth_headers,
                                                      monkeypatch):
    backend.add_plan("t1", "user-1", "{}")
    _enable_langfuse(monkeypatch, _FakeLangfuse(raise_on_score=True))
    resp = await client.post("/v1/feedback/t1", json={"score": 4},
                             headers=auth_headers("user-1"))
    # Mirror is best-effort: the score still lands in Postgres and the API returns 200.
    assert resp.status_code == 200
    assert backend.feedback[0]["score"] == 4
