"""Retrieve node: agentic RAG — similar past plans become few-shot planner examples (v9 stage 2)."""

from __future__ import annotations

import time

import structlog

from app.core.metrics import node_latency_seconds
from app.core.state import AgentState
from app.rag.retriever import retrieve

log = structlog.get_logger(__name__)


async def retrieve_node(state: AgentState) -> AgentState:
    """Populate ``state.retrieved_plans`` via the RAG loop in app/rag/retriever.py.

    Runs BEFORE plan (v9): the retriever searches past successful plans keyed on the
    user request and feeds them to the planner as few-shot examples. The guard's
    replan edge skips this node — retrieved_plans is already populated on a retry.
    """
    with structlog.contextvars.bound_contextvars(
        trace_id=state.trace_id,
        user_id=state.user_id,
        session_id=state.session_id,
        node="retrieve",
    ):
        t0 = time.monotonic()
        try:
            state.retrieved_plans = await retrieve(state)
            log.info("retrieve_done", retrieved=len(state.retrieved_plans))
        except Exception as exc:
            log.exception("retrieve_failed", error=str(exc))
            state.error = f"retrieve_failed:{exc!s}"
        node_latency_seconds.labels(node="retrieve").observe(time.monotonic() - t0)
        return state
