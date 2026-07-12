"""Compose node: merge tool outputs into the user-facing DraftResponse (v9 stage 5)."""

from __future__ import annotations

import time

import structlog

from app.agents.response_composer import compose
from app.core.metrics import node_latency_seconds
from app.core.state import AgentState
from app.data.repositories import save_plan

log = structlog.get_logger(__name__)


async def compose_node(state: AgentState) -> AgentState:
    """Build state.draft from the tool results; skip when execute already set a final draft.

    A non-None draft on entry means execute produced a terminal answer itself (the
    calendar "no free slot" report) — composing over it would erase that message.
    """
    with structlog.contextvars.bound_contextvars(
        trace_id=state.trace_id,
        user_id=state.user_id,
        session_id=state.session_id,
        node="compose",
    ):
        t0 = time.monotonic()
        try:
            if state.draft is None:
                state.draft = await compose(state)
                await save_plan(state)  # persist the draft immediately so the result survives
                log.info("compose_done", tool_results=len(state.tool_results))
            else:
                log.info("compose_skipped_existing_draft")
        except Exception as exc:
            log.exception("compose_failed", error=str(exc))
            state.error = f"compose_failed:{exc!s}"
        node_latency_seconds.labels(node="compose").observe(time.monotonic() - t0)
        return state
