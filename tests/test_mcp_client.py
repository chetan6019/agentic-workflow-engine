"""MCP client tests: idempotency keys/cache, retries, unknown server/tool.

An MCPClient is assembled over a fake tool registry and a dict-backed fake
Redis — no MCP servers, network, or real Redis. ``MCPClient.__new__`` skips the
adapter construction; the preloaded ``_tools_by_server`` registry means
``_load_server_tools`` never touches the (absent) MultiServerMCPClient.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.state import ToolResult
from app.mcp import client as mcp_client
from app.mcp.client import MCPClient, _idempotency_key


class _FakeRedis:
    """Dict-backed stand-in for the async Redis client (decode_responses=True)."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


class _FakeTool:
    """Counts invocations; fails `fail_times` times with `exc`, then returns `result`."""

    def __init__(self, name, result=None, fail_times=0, exc=None):
        self.name = name
        self._result = {"ok": "yes"} if result is None else result
        self._fail_times = fail_times
        self._exc = exc or httpx.TimeoutException("timeout")
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return self._result


def _client(calendar_tools: dict[str, _FakeTool]) -> MCPClient:
    """MCPClient with a preloaded registry and no adapter behind it."""
    c = MCPClient.__new__(MCPClient)
    c._mcp = None  # any fall-through to the adapter fails loudly in tests
    c._servers = {"calendar", "gmail", "notion", "slack"}
    c._tools_by_server = {"calendar": calendar_tools, "gmail": {}, "notion": {}, "slack": {}}
    return c


@pytest.fixture
def fake_redis(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(mcp_client, "get_redis", lambda: r)
    return r


# ── idempotency key ──────────────────────────────────────────────────────────


def test_idempotency_key_is_stable_and_step_scoped():
    k_ab = _idempotency_key("t", "s1", "create_event", {"a": 1, "b": 2})
    k_ba = _idempotency_key("t", "s1", "create_event", {"b": 2, "a": 1})
    k_s2 = _idempotency_key("t", "s2", "create_event", {"a": 1, "b": 2})
    k_t2 = _idempotency_key("t2", "s1", "create_event", {"a": 1, "b": 2})
    assert k_ab == k_ba  # arg order must not matter
    assert k_ab != k_s2  # different step → different key
    assert k_ab != k_t2  # different trace → different key


# ── cache behaviour ──────────────────────────────────────────────────────────


async def test_cache_hit_short_circuits_the_tool(fake_redis):
    tool = _FakeTool("create_event")
    c = _client({"create_event": tool})
    cached = ToolResult(step_id="s1", ok=True, output={"cached": True}, error=None, latency_ms=5)
    key = _idempotency_key("t", "s1", "create_event", {"x": 1})
    fake_redis.store[f"mcp:{key}"] = cached.model_dump_json()

    result = await c.call_tool("calendar", "create_event", {"x": 1}, trace_id="t", step_id="s1")

    assert result.output == {"cached": True}
    assert tool.calls == 0  # tool never invoked on a cache hit


async def test_result_is_cached_and_replay_skips_the_tool(fake_redis):
    tool = _FakeTool("create_event")
    c = _client({"create_event": tool})

    first = await c.call_tool("calendar", "create_event", {"x": 1}, trace_id="t", step_id="s1")
    second = await c.call_tool("calendar", "create_event", {"x": 1}, trace_id="t", step_id="s1")

    assert first.ok and second.ok
    assert tool.calls == 1  # replay served from Redis
    assert any(k.startswith("mcp:") for k in fake_redis.store)


# ── retries ──────────────────────────────────────────────────────────────────


async def test_transient_error_retries_then_succeeds(fake_redis):
    tool = _FakeTool("create_event", fail_times=1)  # one timeout, then success
    c = _client({"create_event": tool})

    result = await c.call_tool("calendar", "create_event", {}, trace_id="t", step_id="s1")

    assert result.ok is True
    assert tool.calls == 2


async def test_nontransient_error_fails_without_retry(fake_redis):
    tool = _FakeTool("create_event", fail_times=99, exc=ValueError("boom"))
    c = _client({"create_event": tool})

    result = await c.call_tool("calendar", "create_event", {}, trace_id="t", step_id="s1")

    assert result.ok is False
    assert "boom" in (result.error or "")
    assert tool.calls == 1  # ValueError is not in _TRANSIENT_EXC → no retry


# ── resolution failures stay honest (ok=False, never raise) ──────────────────


async def test_unknown_tool_returns_failed_result(fake_redis):
    c = _client({})
    result = await c.call_tool("calendar", "nope", {}, trace_id="t", step_id="s1")
    assert result.ok is False
    assert result.error == "unknown_tool:calendar/nope"


async def test_unknown_server_returns_failed_result(fake_redis):
    c = _client({})
    result = await c.call_tool("fax", "beep", {}, trace_id="t", step_id="s1")
    assert result.ok is False
    assert (result.error or "").startswith("unknown_server")


async def test_swapped_server_and_tool_are_resolved(fake_redis):
    """The client tolerates a planner that swaps the server/action fields."""
    tool = _FakeTool("create_event")
    c = _client({"create_event": tool})

    result = await c.call_tool("create_event", "calendar", {}, trace_id="t", step_id="s1")

    assert result.ok is True
    assert tool.calls == 1
