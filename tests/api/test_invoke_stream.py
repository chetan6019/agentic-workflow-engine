"""Async route tests for GET /v1/invoke/stream/{trace_id} (SSE) + web CORS origins.

Same strategy as the other API tests: real app via httpx ASGITransport, conftest's
in-memory Backend behind the routers. Streams are exercised with pre-seeded
terminal/near-terminal states so no test ever sleeps through the poll loop.
"""

from __future__ import annotations

import json

from fastapi.middleware.cors import CORSMiddleware

from app.main import app
from app.security.jwt_tokens import create_access_token

_TERMINAL_STATE = json.dumps({
    "draft": {"detail_markdown": "All done.", "summary": "done",
              "actions_taken": [], "actions_pending": [], "citations": []},
})


def _phase_payload(owner: str, *, phase: str = "💾 finalizing", done: bool = True) -> str:
    return json.dumps({"phase": phase, "done": done, "owner": owner})


async def _stream_body(client, url: str, headers: dict | None = None) -> tuple[int, str]:
    """GET the SSE endpoint and return (status_code, full streamed body)."""
    async with client.stream("GET", url, headers=headers or {}) as resp:
        chunks = [chunk async for chunk in resp.aiter_text()]
        return resp.status_code, "".join(chunks)


async def test_stream_401_without_any_token(client, backend):
    status, _ = await _stream_body(client, "/v1/invoke/stream/t1")
    assert status == 401


async def test_stream_query_param_token_authenticates(client, backend):
    # Terminal plan row → the pre-loop shortcut answers with one result event.
    backend.add_plan("t1", user_id="user-1", state_json=_TERMINAL_STATE)
    token = create_access_token("user-1")
    status, body = await _stream_body(client, f"/v1/invoke/stream/t1?access_token={token}")
    assert status == 200
    assert "event: result" in body
    assert '"status": "done"' in body


async def test_stream_header_auth_still_works(client, backend, auth_headers):
    backend.add_plan("t1", user_id="user-1", state_json=_TERMINAL_STATE)
    status, body = await _stream_body(client, "/v1/invoke/stream/t1",
                                      headers=auth_headers("user-1"))
    assert status == 200
    assert "event: result" in body


async def test_stream_emits_phase_then_result_from_loop(client, backend, auth_headers):
    # No plan row yet (run just started) → pre-loop is "pending" and the loop runs.
    # The phase key is already done, so the loop emits one phase frame then result.
    backend.redis.kv["phase:t2"] = _phase_payload("user-1")
    status, body = await _stream_body(client, "/v1/invoke/stream/t2",
                                      headers=auth_headers("user-1"))
    assert status == 200
    assert "event: phase" in body
    # json.dumps escapes the emoji on the wire; assert on the label text instead.
    assert "finalizing" in body
    assert "event: result" in body


async def test_stream_cross_user_plan_is_404(client, backend, auth_headers):
    # Ownership mismatch found BEFORE streaming starts → a real 404, not an SSE frame.
    backend.add_plan("t3", user_id="owner-2", state_json=_TERMINAL_STATE)
    status, _ = await _stream_body(client, "/v1/invoke/stream/t3",
                                   headers=auth_headers("user-1"))
    assert status == 404


async def test_stream_cross_user_phase_key_emits_error_event(client, backend, auth_headers):
    # Mismatch discovered mid-stream (phase key owned by someone else, no plan row):
    # the response is already 200/streaming, so it ends with an error event.
    backend.redis.kv["phase:t4"] = _phase_payload("owner-2")
    status, body = await _stream_body(client, "/v1/invoke/stream/t4",
                                      headers=auth_headers("user-1"))
    assert status == 200
    assert "event: error" in body
    assert "not_found" in body


async def test_stream_result_immediate_when_phase_key_expired(client, backend, auth_headers):
    # Terminal plan row but the 600s phase key is gone (connect-after-completion):
    # the pre-loop shortcut must answer immediately instead of spinning to timeout.
    backend.add_plan("t5", user_id="user-1", state_json=_TERMINAL_STATE)
    status, body = await _stream_body(client, "/v1/invoke/stream/t5",
                                      headers=auth_headers("user-1"))
    assert status == 200
    assert "event: result" in body
    assert "event: phase" not in body


async def test_workflow_pause_writes_awaiting_approval_phase(backend, monkeypatch):
    """A run that interrupts for HITL must surface '🛑 awaiting approval' with
    done=true (same wire behavior as before the native-interrupt migration) and
    must NOT persist the pre-pause state, save a message, or discard the thread."""
    from app.api import invoke as invoke_mod
    from app.core.state import AgentState

    class FakePausingCompiled:
        async def astream(self, initial, config=None, stream_mode=None):
            yield initial.model_dump()
            yield {"__interrupt__": (object(),)}

    saved: list = []
    discarded: list = []

    async def fake_save_plan(state):
        saved.append(state)

    async def fake_discard(trace_id):
        discarded.append(trace_id)

    monkeypatch.setattr(invoke_mod, "compile_graph", lambda: FakePausingCompiled())
    monkeypatch.setattr(invoke_mod, "save_plan", fake_save_plan)
    monkeypatch.setattr(invoke_mod, "discard_thread", fake_discard)

    state = AgentState(trace_id="t-pause", user_id="user-1", session_id="s",
                       user_request="do a thing")
    await invoke_mod._run_workflow_with_phase(state)

    phase = json.loads(backend.redis.kv["phase:t-pause"])
    assert phase["done"] is True
    assert "awaiting approval" in phase["phase"]
    assert saved == []       # the paused node persisted its own state; don't clobber
    assert discarded == []   # the checkpoint is the resume source — keep it


def test_cors_allows_web_origins():
    cors = next(m for m in app.user_middleware if m.cls is CORSMiddleware)
    origins = cors.kwargs["allow_origins"]
    assert "http://localhost:8501" in origins   # Streamlit stays allowed
    assert "http://localhost:5173" in origins   # Vite dev server
    assert "http://localhost:8502" in origins   # nginx-served prod build
