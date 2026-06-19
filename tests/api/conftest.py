"""Shared fixtures for the async API route tests.

Strategy (per the approved plan): real FastAPI app exercised through
httpx.AsyncClient(ASGITransport), with the DB / Redis / graph / background spawn
faked at each router's import boundary — no Docker, runs in bare CI. Auth uses a
REAL HS256 JWT minted with the app's own secret, so the real middleware decodes it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import approvals as approvals_mod
from app.api import feedback as feedback_mod
from app.api import integrations as integrations_mod
from app.api import invoke as invoke_mod
from app.guardrails import engine as guard_engine
from app.main import app
from app.security.jwt_tokens import create_access_token

import app.main as main_mod


class FakeRedis:
    """Minimal async Redis stand-in: kv (get/set) + counters (incr/incrby/expire)."""

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.counters: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self.kv.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.kv[key] = value

    async def incr(self, key: str) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    async def incrby(self, key: str, amount: int) -> int:
        self.counters[key] = self.counters.get(key, 0) + amount
        return self.counters[key]

    async def expire(self, key: str, ttl: int) -> None:
        return None


class Backend:
    """In-memory store backing the faked repository functions."""

    def __init__(self) -> None:
        self.plans: dict[str, dict[str, Any]] = {}
        self.approvals: dict[str, dict[str, Any]] = {}
        self.tokens: dict[tuple[str, str], str] = {}
        self.feedback: list[dict[str, Any]] = []
        self.spawned: list[Any] = []
        self.resumed: list[Any] = []
        self.redis = FakeRedis()

    # convenience seeders -----------------------------------------------------
    def add_plan(self, trace_id: str, user_id: str, state_json: str = "{}") -> None:
        self.plans[trace_id] = {"trace_id": trace_id, "user_id": user_id, "state_json": state_json}

    def add_approval(self, token: str, trace_id: str, *, hours: float = 1.0,
                     decision: str | None = None) -> None:
        self.approvals[token] = {"token": token, "trace_id": trace_id, "decision": decision,
                                 "expires_at": datetime.now(timezone.utc) + timedelta(hours=hours)}

    def add_token(self, user_id: str, provider: str, token_enc: str) -> None:
        self.tokens[(user_id, provider)] = token_enc


@pytest.fixture
def backend(monkeypatch) -> Backend:
    """Wire every router's data/redis/graph/spawn collaborators to one in-memory Backend."""
    b = Backend()

    # ---- invoke -------------------------------------------------------------
    async def create_session(user_id: str) -> str:
        return "sess-1"

    async def get_recent_messages(session_id: str, limit: int) -> list[dict[str, str]]:
        return []

    async def save_message(*a: Any, **k: Any) -> None:
        return None

    def fake_spawn(coro: Any) -> None:
        b.spawned.append(coro)
        coro.close()  # never actually run the graph in a route test

    async def get_plan(trace_id: str) -> dict[str, Any] | None:
        return b.plans.get(trace_id)

    monkeypatch.setattr(invoke_mod, "create_session", create_session)
    monkeypatch.setattr(invoke_mod, "get_recent_messages", get_recent_messages)
    monkeypatch.setattr(invoke_mod, "save_message", save_message)
    monkeypatch.setattr(invoke_mod, "spawn", fake_spawn)
    monkeypatch.setattr(invoke_mod, "get_redis", lambda: b.redis)
    monkeypatch.setattr(invoke_mod, "get_plan_by_trace_id", get_plan)
    monkeypatch.setattr(guard_engine, "get_redis", lambda: b.redis)

    # ---- middleware (no real Redis rate limiting unless a test overrides) ----
    async def allow_rate(identity: str, limit: int, window_sec: int = 60) -> bool:
        return True

    monkeypatch.setattr(main_mod, "check_rate_limit", allow_rate)

    # ---- approvals ----------------------------------------------------------
    async def get_approval(token: str) -> dict[str, Any] | None:
        return b.approvals.get(token)

    async def get_pending(user_id: str) -> list[dict[str, Any]]:
        return [a for a in b.approvals.values()
                if a["decision"] is None and b.plans.get(a["trace_id"], {}).get("user_id") == user_id]

    async def save_plan(state: Any) -> None:
        b.resumed.append(state)

    async def update_status(token: str, decision: str) -> None:
        if token in b.approvals:
            b.approvals[token]["decision"] = decision

    class FakeCompiled:
        async def ainvoke(self, state: Any, config: Any = None) -> Any:
            b.resumed.append(state)
            return state.model_dump()

    async def discard(trace_id: str) -> None:
        return None

    monkeypatch.setattr(approvals_mod, "get_approval_by_token", get_approval)
    monkeypatch.setattr(approvals_mod, "get_pending_approvals_by_user", get_pending)
    monkeypatch.setattr(approvals_mod, "get_plan_by_trace_id", get_plan)
    monkeypatch.setattr(approvals_mod, "save_plan", save_plan)
    monkeypatch.setattr(approvals_mod, "update_approval_status", update_status)
    monkeypatch.setattr(approvals_mod, "compile_graph", lambda: FakeCompiled())
    monkeypatch.setattr(approvals_mod, "discard_thread", discard)

    # ---- feedback -----------------------------------------------------------
    async def save_feedback(*, trace_id: str, score: int, comment: str | None = None) -> None:
        b.feedback.append({"trace_id": trace_id, "score": score, "comment": comment})

    monkeypatch.setattr(feedback_mod, "get_plan_by_trace_id", get_plan)
    monkeypatch.setattr(feedback_mod, "save_feedback", save_feedback)
    monkeypatch.setattr(feedback_mod, "check_rate_limit", allow_rate)

    # ---- integrations -------------------------------------------------------
    async def get_user_tokens(user_id: str) -> list[dict[str, Any]]:
        return [{"provider": p, "token_enc": enc}
                for (uid, p), enc in b.tokens.items() if uid == user_id]

    async def save_token(*, user_id: str, provider: str, token_enc: str) -> None:
        b.tokens[(user_id, provider)] = token_enc

    async def delete_token(user_id: str, provider: str) -> bool:
        return b.tokens.pop((user_id, provider), None) is not None

    monkeypatch.setattr(integrations_mod, "get_user_tokens", get_user_tokens)
    monkeypatch.setattr(integrations_mod, "save_token", save_token)
    monkeypatch.setattr(integrations_mod, "delete_token", delete_token)
    return b


@pytest.fixture
async def client() -> Any:
    """httpx.AsyncClient bound to the real ASGI app (true async route semantics)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Factory: auth_headers(user_id) -> a real Bearer JWT signed with the app secret."""
    def _make(user_id: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {create_access_token(user_id)}"}

    return _make
