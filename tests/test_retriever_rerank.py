"""Cross-encoder rerank unit tests: ordering, truncation, off-by-default, retrieve() wiring.

These exercise ``app.rag.retriever._rerank`` and ``retrieve`` with a stubbed
``rerank_scores`` and stubbed settings, so they run with no fastembed model download,
no network, and no Qdrant.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.rag import retriever


def _plan(pid: str, fused: float = 0.0, sim: float = 0.5) -> retriever.RetrievedPlan:
    """A minimal RetrievedPlan candidate."""
    return retriever.RetrievedPlan(
        plan_id=pid, request_text=f"req-{pid}", plan_json={},
        summary=f"sum-{pid}", similarity=sim, fused_score=fused,
    )


def _set_rerank(monkeypatch, enabled: bool) -> None:
    """Stub settings so only the rerank toggle matters (search/grade are stubbed away)."""
    monkeypatch.setattr(
        retriever, "get_settings",
        lambda: SimpleNamespace(rerank_enabled=enabled, hybrid_search_enabled=True),
    )


async def test_rerank_reorders_by_cross_encoder_score(monkeypatch):
    """When enabled, candidates come back ordered by the cross-encoder score, highest first."""
    _set_rerank(monkeypatch, True)

    async def _scores(_query, docs):
        # Ascending score with index → the LAST doc is most relevant.
        return [float(i) for i in range(len(docs))]

    monkeypatch.setattr(retriever, "rerank_scores", _scores)
    cands = [_plan(str(i)) for i in range(4)]

    out = await retriever._rerank("q", cands)

    assert [p.plan_id for p in out] == ["3", "2", "1", "0"]


async def test_rerank_truncates_to_top_n(monkeypatch):
    """More candidates than _RERANK_TOP_N → only the top _RERANK_TOP_N survive."""
    _set_rerank(monkeypatch, True)

    async def _scores(_query, docs):
        return [float(i) for i in range(len(docs))]

    monkeypatch.setattr(retriever, "rerank_scores", _scores)
    cands = [_plan(str(i)) for i in range(8)]

    out = await retriever._rerank("q", cands)

    assert len(out) == retriever._RERANK_TOP_N
    assert [p.plan_id for p in out] == ["7", "6", "5", "4", "3"]


async def test_rerank_is_noop_when_disabled(monkeypatch):
    """Toggle off → input returned unchanged and the encoder is never invoked."""
    _set_rerank(monkeypatch, False)
    called: list[int] = []

    async def _scores(_query, docs):
        called.append(1)
        return [0.0] * len(docs)

    monkeypatch.setattr(retriever, "rerank_scores", _scores)
    cands = [_plan("a"), _plan("b")]

    out = await retriever._rerank("q", cands)

    assert out == cands
    assert called == []


async def test_retrieve_preserves_rerank_order_over_fused(monkeypatch):
    """With rerank on, retrieve() keeps the rerank order instead of re-sorting by fused_score."""
    _set_rerank(monkeypatch, True)

    async def _route(_state, _meta):
        return retriever._RouteDecision(should_retrieve=True, query="q")

    # fused_score ascends with plan_id, so a fused sort would yield 4,3,2; the reranker
    # keeps input order (0,1,2,...), proving rerank order wins over the fused score.
    cands = [_plan(str(i), fused=float(i)) for i in range(5)]

    async def _search_stub(_query, _filters, k=10):
        return cands

    async def _grade_stub(c, _req, _meta):
        return c

    async def _scores(_query, docs):
        # Descending score with index → keep the incoming order.
        return [float(len(docs) - i) for i in range(len(docs))]

    monkeypatch.setattr(retriever, "_route_query", _route)
    monkeypatch.setattr(retriever, "_search", _search_stub)
    monkeypatch.setattr(retriever, "_grade", _grade_stub)
    monkeypatch.setattr(retriever, "rerank_scores", _scores)

    state = retriever.AgentState(trace_id="t", user_id="u", session_id="s", user_request="q")
    out = await retriever.retrieve(state)

    assert [p.plan_id for p in out] == ["0", "1", "2"]  # rerank order, not fused (4,3,2)


# Langfuse retrieval-event tests removed with the helper they covered — the OTel
# "retrieve" span carries the funnel now (see tests/test_tracing.py).
