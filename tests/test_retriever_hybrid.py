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

    def query_points(self, **kwargs):
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
    async def _yes(_state):
        return True

    async def _rw(req):
        return req

    candidates = [
        retriever.RetrievedPlan(plan_id=str(i), request_text="r", plan_json={},
                                summary="s", similarity=0.9 - i * 0.01,
                                fused_score=float(i))
        for i in range(7)
    ]

    async def _search_stub(_query, _filters, k=10):
        return candidates

    async def _grade_stub(cands, _req):
        return cands

    monkeypatch.setattr(retriever, "_should_retrieve", _yes)
    monkeypatch.setattr(retriever, "_rewrite_query", _rw)
    monkeypatch.setattr(retriever, "_search", _search_stub)
    monkeypatch.setattr(retriever, "_grade", _grade_stub)

    state = retriever.AgentState(
        trace_id="t", user_id="u", session_id="s", user_request="do a thing"
    )
    out = await retriever.retrieve(state)

    assert len(out) == retriever._TOP_N
    assert [p.plan_id for p in out] == ["6", "5", "4", "3", "2"]  # highest fused first
