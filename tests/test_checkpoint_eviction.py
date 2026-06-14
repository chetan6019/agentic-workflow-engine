"""discard_thread tests (REVIEW.md R35): evicts the checkpoint, fails open."""

from __future__ import annotations

from types import SimpleNamespace

from app.orchestration import graph


async def test_discard_thread_calls_adelete(monkeypatch):
    deleted: list[str] = []

    async def fake_adelete(trace_id):
        deleted.append(trace_id)

    fake_compiled = SimpleNamespace(checkpointer=SimpleNamespace(adelete_thread=fake_adelete))
    monkeypatch.setattr(graph, "compile_graph", lambda: fake_compiled)

    await graph.discard_thread("trace-123")
    assert deleted == ["trace-123"]


async def test_discard_thread_swallows_errors(monkeypatch):
    async def boom(trace_id):
        raise RuntimeError("checkpointer gone")

    fake_compiled = SimpleNamespace(checkpointer=SimpleNamespace(adelete_thread=boom))
    monkeypatch.setattr(graph, "compile_graph", lambda: fake_compiled)

    await graph.discard_thread("trace-123")  # must not raise
