"""Hybrid-retrieval unit tests: RRF ordering, cosine preservation, toggle fallback.

These exercise ``app.rag.retriever`` with a fake Qdrant client and stubbed embedders,
so they run with no network, no Qdrant, and no fastembed model download.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client.http.models import SparseVector

from app.rag import retriever


def _hit(pid: str, score: float):
    """A minimal stand-in for a Qdrant scored point."""
    return SimpleNamespace(
        id=pid,
        score=score,
        payload={"request_text": f"req-{pid}", "plan_json": {}, "summary": f"sum-{pid}"},
    )


class _FakeClient:
    """Returns fused points when prefetch is present, else dense points."""

    def __init__(self, fused, dense):
        self._fused = fused
        self._dense = dense
        self.calls: list[str] = []

    async def query_points(self, **kwargs):
        if "prefetch" in kwargs:
            self.calls.append("fused")
            return SimpleNamespace(points=self._fused)
        self.calls.append("dense")
        return SimpleNamespace(points=self._dense)


@pytest.fixture
def patch_embedders(monkeypatch):
    """Stub both embedders so no model/network is touched."""
    async def _dense(_text):
        return [0.1] * 768

    async def _sparse(_text):
        return SparseVector(indices=[1, 2], values=[1.0, 1.0])

    monkeypatch.setattr(retriever, "embed_text", _dense)
    monkeypatch.setattr(retriever, "embed_sparse", _sparse)


def _set_hybrid(monkeypatch, enabled: bool):
    monkeypatch.setattr(
        retriever, "get_settings", lambda: SimpleNamespace(hybrid_search_enabled=enabled)
    )


def test_to_plan_clamps_similarity_into_cosine_range():
    high = retriever._to_plan(_hit("A", 0.0), similarity=1.4, fused_score=0.0)
    low = retriever._to_plan(_hit("B", 0.0), similarity=-0.2, fused_score=0.0)
    assert high is not None and high.similarity == 1.0
    assert low is not None and low.similarity == 0.0


async def test_hybrid_search_preserves_cosine_and_uses_fused_order(monkeypatch, patch_embedders):
    """Fused RRF order drives the list; similarity is the dense cosine, not the RRF score."""
    _set_hybrid(monkeypatch, True)
    # Fusion ranks B above A; dense cosine is the opposite (A more similar).
    fused = [_hit("B", 0.033), _hit("A", 0.016)]
    dense = [_hit("A", 0.91), _hit("B", 0.50)]
    fake = _FakeClient(fused, dense)
    monkeypatch.setattr(retriever, "get_qdrant", lambda: fake)

    plans = await retriever._search("find a meeting", {"user_id": "u1", "success": True})

    assert [p.plan_id for p in plans] == ["B", "A"]  # fused order
    by_id = {p.plan_id: p for p in plans}
    assert by_id["A"].similarity == pytest.approx(0.91)  # cosine, from the dense pass
    assert by_id["B"].similarity == pytest.approx(0.50)
    assert by_id["B"].fused_score == pytest.approx(0.033)  # RRF score kept separate
    assert "fused" in fake.calls and "dense" in fake.calls


async def test_dense_only_fallback_when_hybrid_disabled(monkeypatch, patch_embedders):
    """Toggle off → single dense pass; fused_score mirrors the cosine so sorting still works."""
    _set_hybrid(monkeypatch, False)
    dense = [_hit("A", 0.91), _hit("B", 0.50)]
    fake = _FakeClient(fused=[], dense=dense)
    monkeypatch.setattr(retriever, "get_qdrant", lambda: fake)

    plans = await retriever._search("find a meeting", {"success": True})

    assert fake.calls == ["dense"]  # no fused query ever issued
    assert [p.plan_id for p in plans] == ["A", "B"]
    for p in plans:
        assert p.fused_score == p.similarity


async def test_retrieve_orders_by_fused_score(monkeypatch):
    """retrieve() must rank by fused_score, not cosine, and cap at the top-N."""
    async def _route(_state, _meta):
        return retriever._RouteDecision(should_retrieve=True, query="do a thing")

    candidates = [
        retriever.RetrievedPlan(plan_id=str(i), request_text="r", plan_json={},
                                summary="s", similarity=0.9 - i * 0.01,
                                fused_score=float(i))
        for i in range(7)
    ]

    async def _search_stub(_query, _filters, k=10):
        return candidates

    async def _grade_stub(cands, _req, _meta):
        return cands

    monkeypatch.setattr(retriever, "_route_query", _route)
    monkeypatch.setattr(retriever, "_search", _search_stub)
    monkeypatch.setattr(retriever, "_grade", _grade_stub)

    state = retriever.AgentState(
        trace_id="t", user_id="u", session_id="s", user_request="do a thing"
    )
    out = await retriever.retrieve(state)

    assert len(out) == retriever._TOP_N
    # Full fused-descending order is 6..0; retrieve() returns the first _TOP_N of it.
    full_fused_order = ["6", "5", "4", "3", "2", "1", "0"]
    assert [p.plan_id for p in out] == full_fused_order[: retriever._TOP_N]


async def test_route_query_falls_back_to_raw_request_on_llm_failure(monkeypatch):
    """Router LLM failure must degrade to retrieving with the raw request."""
    class _Boom:
        async def ainvoke(self, _msgs):
            raise RuntimeError("llm down")

    monkeypatch.setattr(retriever, "get_structured_llm", lambda *a, **k: _Boom())
    state = retriever.AgentState(trace_id="t", user_id="u", session_id="s",
                                 user_request="find my unread emails")

    route = await retriever._route_query(state, {})

    assert route.should_retrieve is True
    assert route.query == "find my unread emails"


async def test_retrieve_skips_search_when_router_says_no(monkeypatch):
    """should_retrieve=False must short-circuit before any Qdrant search."""
    async def _route(_state, _meta):
        return retriever._RouteDecision(should_retrieve=False, query="q")

    searched: list[int] = []

    async def _search_stub(*_a, **_k):
        searched.append(1)
        return []

    monkeypatch.setattr(retriever, "_route_query", _route)
    monkeypatch.setattr(retriever, "_search", _search_stub)
    state = retriever.AgentState(trace_id="t", user_id="u", session_id="s",
                                 user_request="hello there")

    assert await retriever.retrieve(state) == []
    assert searched == []


def test_estimate_tokens_counts_payload_text():
    plan = retriever.RetrievedPlan(
        plan_id="p1", request_text="a" * 40, summary="b" * 40,
        plan_json={}, similarity=0.9, fused_score=0.9)
    # 40 + 40 chars of text + 2 chars of "{}" = 82 chars → 82 // 4 = 20 tokens.
    assert retriever._estimate_tokens([plan]) == 20
    assert retriever._estimate_tokens([]) == 0


class _FakeRedis:
    """Minimal async get/set stand-in for the router cache."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


def _patch_router_cache(monkeypatch, ttl: int = 3600) -> _FakeRedis:
    fake = _FakeRedis()
    monkeypatch.setattr(retriever, "get_redis", lambda: fake)
    monkeypatch.setattr(retriever, "get_settings",
                        lambda: SimpleNamespace(router_cache_ttl_sec=ttl))
    return fake


async def test_route_query_caches_decision_and_skips_llm_on_repeat(monkeypatch):
    """Second call with the same (normalized) request must not touch the LLM."""
    calls: list[int] = []

    class _LLM:
        async def ainvoke(self, _msgs):
            calls.append(1)
            return retriever._RouteDecision(should_retrieve=True, query="unread emails")

    fake_redis = _patch_router_cache(monkeypatch)
    monkeypatch.setattr(retriever, "get_structured_llm", lambda *a, **k: _LLM())

    first = retriever.AgentState(trace_id="t1", user_id="u", session_id="s",
                                 user_request="Show my unread emails")
    # Different casing/whitespace — must normalize to the same cache key.
    second = retriever.AgentState(trace_id="t2", user_id="u", session_id="s",
                                  user_request="show  my UNREAD emails")

    r1 = await retriever._route_query(first, {})
    r2 = await retriever._route_query(second, {})

    assert len(calls) == 1  # one LLM call total; second was a cache hit
    assert r1.query == r2.query == "unread emails"
    assert len(fake_redis.store) == 1


async def test_route_query_fallback_is_never_cached(monkeypatch):
    """An LLM failure must not poison the cache with the raw-request fallback."""
    class _Boom:
        async def ainvoke(self, _msgs):
            raise RuntimeError("llm down")

    fake_redis = _patch_router_cache(monkeypatch)
    monkeypatch.setattr(retriever, "get_structured_llm", lambda *a, **k: _Boom())
    state = retriever.AgentState(trace_id="t", user_id="u", session_id="s",
                                 user_request="find my unread emails")

    route = await retriever._route_query(state, {})

    assert route.should_retrieve is True
    assert fake_redis.store == {}


async def test_route_query_cache_disabled_by_zero_ttl(monkeypatch):
    """router_cache_ttl_sec=0 must bypass Redis entirely."""
    calls: list[int] = []

    class _LLM:
        async def ainvoke(self, _msgs):
            calls.append(1)
            return retriever._RouteDecision(should_retrieve=True, query="q")

    fake_redis = _patch_router_cache(monkeypatch, ttl=0)
    monkeypatch.setattr(retriever, "get_structured_llm", lambda *a, **k: _LLM())
    state = retriever.AgentState(trace_id="t", user_id="u", session_id="s",
                                 user_request="find my unread emails")

    await retriever._route_query(state, {})
    await retriever._route_query(state, {})

    assert len(calls) == 2  # no caching
    assert fake_redis.store == {}
