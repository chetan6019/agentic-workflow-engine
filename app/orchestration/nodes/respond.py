"""Respond node: persist the finished run and index its plan into Qdrant (v9 stage 7)."""

from __future__ import annotations

import time

import structlog

from app.core.background import spawn
from app.core.metrics import node_latency_seconds
from app.core.state import AgentState
from app.data.repositories import save_plan
from app.rag.indexer import index_plan

log = structlog.get_logger(__name__)


async def respond_node(state: AgentState) -> AgentState:
    """Persist the final plan and schedule background indexing of the finished run."""
    with structlog.contextvars.bound_contextvars(
        trace_id=state.trace_id,
        user_id=state.user_id,
        session_id=state.session_id,
        node="respond",
    ):
        t0 = time.monotonic()
        try:
            await save_plan(state)
            spawn(index_plan(state))  # tracked so it survives GC and drains on shutdown
            log.info("respond_done", confidence=round(state.confidence, 3))
        except Exception as exc:
            log.exception("respond_failed", error=str(exc))
            state.error = f"respond_failed:{exc!s}"
        node_latency_seconds.labels(node="respond").observe(time.monotonic() - t0)
        return state
