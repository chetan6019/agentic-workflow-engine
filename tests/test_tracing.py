"""OTel tracing tests: no-op bootstrap, span emission, and structlog correlation.

An in-memory span exporter stands in for the collector — no OTLP endpoint,
network, or Alloy involved. The provider is registered once at import time
(OTel allows exactly one global provider per process); app modules created
their tracers via ``trace.get_tracer`` which defers to this provider lazily.
"""

from __future__ import annotations

import httpx

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core.tracing import setup_tracing
from app.logging import add_otel_trace_context
from app.mcp import client as mcp_client
from app.mcp.client import MCPClient

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)
_test_tracer = trace.get_tracer(__name__)


# ── bootstrap ────────────────────────────────────────────────────────────────


def test_setup_tracing_is_noop_without_endpoint(monkeypatch):
    class _Settings:
        otel_exporter_otlp_endpoint = None
        otel_service_name = "api"

    monkeypatch.setattr("app.core.tracing.get_settings", lambda: _Settings())
    # Must return without touching the app or the global provider.
    setup_tracing(app=None)  # type: ignore[arg-type]  # never dereferenced on the no-op path
    assert trace.get_tracer_provider() is _provider  # unchanged


# ── MCP tool-call span ───────────────────────────────────────────────────────
# Fakes mirror tests/test_mcp_client.py: dict-backed Redis, preloaded registry.


class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


class _FakeTool:
    def __init__(self, name, fail_times=0):
        self.name = name
        self._fail_times = fail_times
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise httpx.TimeoutException("timeout")
        return {"ok": "yes"}


def _client(tools: dict[str, _FakeTool]) -> MCPClient:
    c = MCPClient.__new__(MCPClient)
    c._mcp = None
    c._servers = {"google", "github", "reddit", "finnhub"}
    c._tools_by_server = {"google": tools, "github": {}, "reddit": {}, "finnhub": {}}
    return c


async def test_tool_call_emits_span_with_attributes(monkeypatch):
    monkeypatch.setattr(mcp_client, "get_redis", lambda: _FakeRedis())
    _exporter.clear()
    c = _client({"send_email": _FakeTool("send_email")})

    result = await c.call_tool("google", "send_email", {"to": "a@b.c"},
                               trace_id="t", step_id="s1")

    assert result.ok is True
    spans = [s for s in _exporter.get_finished_spans() if s.name == "mcp.tool_call"]
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs["mcp.server"] == "google"
    assert attrs["mcp.tool"] == "send_email"
    assert attrs["mcp.ok"] is True
    assert attrs["mcp.retries"] == 0
    # Identifiers only — tool args must never land on the span (PII).
    assert "a@b.c" not in str(attrs)


async def test_tool_call_span_records_retries_and_failure(monkeypatch):
    monkeypatch.setattr(mcp_client, "get_redis", lambda: _FakeRedis())
    _exporter.clear()
    c = _client({"send_email": _FakeTool("send_email", fail_times=99)})

    result = await c.call_tool("google", "send_email", {}, trace_id="t", step_id="s1")

    assert result.ok is False
    spans = [s for s in _exporter.get_finished_spans() if s.name == "mcp.tool_call"]
    attrs = spans[0].attributes or {}
    assert attrs["mcp.ok"] is False
    assert attrs["mcp.retries"] == 2  # 3 attempts = 2 retries


# ── retriever stage span ─────────────────────────────────────────────────────


async def test_rerank_emits_stage_span(monkeypatch):
    from app.core.state import RetrievedPlan
    from app.rag import retriever

    class _Settings:
        rerank_enabled = True

    monkeypatch.setattr(retriever, "get_settings", lambda: _Settings())

    async def fake_scores(query, docs):
        return [0.9, 0.1]

    monkeypatch.setattr(retriever, "rerank_scores", fake_scores)
    candidates = [
        RetrievedPlan(plan_id="p1", request_text="r1", plan_json={}, summary="s1",
                      similarity=0.8, fused_score=0.8),
        RetrievedPlan(plan_id="p2", request_text="r2", plan_json={}, summary="s2",
                      similarity=0.7, fused_score=0.7),
    ]
    _exporter.clear()

    kept = await retriever._rerank("query", candidates)

    assert kept[0].plan_id == "p1"
    spans = [s for s in _exporter.get_finished_spans() if s.name == "retrieve.rerank"]
    assert len(spans) == 1
    assert (spans[0].attributes or {})["rerank.candidates"] == 2


async def test_retrieve_emits_parent_span_with_funnel_counts(monkeypatch):
    from types import SimpleNamespace

    from app.core.state import RetrievedPlan
    from app.rag import retriever

    monkeypatch.setattr(retriever, "get_settings",
                        lambda: SimpleNamespace(rerank_enabled=False,
                                                hybrid_search_enabled=True))

    async def _route(_state, _meta):
        return retriever._RouteDecision(should_retrieve=True, query="q")

    plans = [RetrievedPlan(plan_id=str(i), request_text="r", plan_json={}, summary="s",
                           similarity=0.9, fused_score=0.9) for i in range(4)]

    async def _search_stub(_query, _filters, k=10):
        return plans

    async def _grade_stub(c, _req, _meta):
        return c

    monkeypatch.setattr(retriever, "_route_query", _route)
    monkeypatch.setattr(retriever, "_search", _search_stub)
    monkeypatch.setattr(retriever, "_grade", _grade_stub)
    _exporter.clear()

    state = retriever.AgentState(trace_id="t", user_id="u", session_id="s", user_request="q")
    kept = await retriever.retrieve(state)

    assert len(kept) == 3  # _TOP_N
    spans = [s for s in _exporter.get_finished_spans() if s.name == "retrieve"]
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs["retrieve.fetched"] == 4
    assert attrs["retrieve.kept"] == 3
    assert attrs["run.trace_id"] == "t"


# ── structlog correlation ────────────────────────────────────────────────────


def test_log_processor_adds_otel_ids_inside_span():
    with _test_tracer.start_as_current_span("unit"):
        event = add_otel_trace_context(None, "info", {"event": "x"})
    assert len(event["otel_trace_id"]) == 32
    assert len(event["otel_span_id"]) == 16


def test_log_processor_is_silent_without_span():
    event = add_otel_trace_context(None, "info", {"event": "x"})
    assert "otel_trace_id" not in event
    assert "otel_span_id" not in event
