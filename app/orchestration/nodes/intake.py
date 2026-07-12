"""Intake node: load session context before retrieval begins (v9 stage 1)."""

from __future__ import annotations

import time

import structlog

from app.core.metrics import node_latency_seconds
from app.core.state import AgentState
from app.data.repositories import get_session

log = structlog.get_logger(__name__)


async def intake_node(state: AgentState) -> AgentState:
    """Load the chat session row so every downstream node runs with known context.

    Input guardrails are enforced synchronously at the API boundary (app/api/invoke.py)
    so a blocked/rate-limited request fails fast with the right HTTP status and the
    graph only ever sees the vetted, PII-redacted ``user_request``.
    """
    with structlog.contextvars.bound_contextvars(
        trace_id=state.trace_id,
        user_id=state.user_id,
        session_id=state.session_id,
        node="intake",
    ):
        t0 = time.monotonic()
        try:
            await get_session(state.session_id)
            log.info("intake_done")
        except Exception as exc:
            log.exception("intake_failed", error=str(exc))
            state.error = f"intake_failed:{exc!s}"
        node_latency_seconds.labels(node="intake").observe(time.monotonic() - t0)
        return state
