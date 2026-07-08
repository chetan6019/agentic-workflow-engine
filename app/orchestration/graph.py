"""LangGraph wiring: StateGraph nodes, conditional edges, and compiled singleton."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import structlog
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.core.state import AgentState
from app.orchestration.nodes.guard import guardrails_node
from app.orchestration.nodes.orchestrator import orchestrator_node
from app.orchestration.nodes.planner import planner_node

log = structlog.get_logger(__name__)

# Module-level checkpointer. init_checkpointer() (called from the FastAPI lifespan)
# swaps in the Postgres saver when CHECKPOINTER_BACKEND=postgres; until then
# MemorySaver keeps tests and bare scripts working with zero setup.
_checkpointer: BaseCheckpointSaver = MemorySaver()
# Holds the AsyncPostgresSaver context manager so shutdown can close its pool.
_saver_ctx: Any = None


async def init_checkpointer() -> None:
    """Open the durable Postgres checkpointer when configured; no-op on memory.

    Must run AFTER init_db() (Postgres reachable). setup() creates LangGraph's own
    checkpoint tables idempotently — they are deliberately outside Alembic because
    the library owns their schema and migrations.
    """
    global _checkpointer, _saver_ctx
    if get_settings().checkpointer_backend != "postgres":
        log.info("checkpointer_memory")
        return
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    # psycopg wants a plain postgresql:// DSN — strip SQLAlchemy's +asyncpg marker
    # (same trick as app/data/db.py's raw-driver path).
    dsn = get_settings().database_url.replace("+asyncpg", "")
    _saver_ctx = AsyncPostgresSaver.from_conn_string(dsn)
    saver = await _saver_ctx.__aenter__()
    await saver.setup()
    _checkpointer = saver
    compile_graph.cache_clear()  # recompile against the durable saver
    log.info("checkpointer_postgres_ready")


async def close_checkpointer() -> None:
    """Close the Postgres saver's pool on shutdown; best-effort, never raises."""
    global _saver_ctx
    if _saver_ctx is None:
        return
    try:
        await _saver_ctx.__aexit__(None, None, None)
    except Exception as exc:
        log.warning("checkpointer_close_failed", error=str(exc))
    _saver_ctx = None


def _route_after_orchestrator(state: AgentState) -> Literal["planner", "guardrails", "__end__"]:
    """Route to planner before a plan exists, to guardrails once an unscored draft exists.

    Pure: reads state only. ``verdict is None`` means guardrails hasn't run yet, so a
    fresh draft is sent there to be scored; once a verdict exists the run is done.
    """
    if state.error:
        dest: str = END
    elif state.plan is None:
        dest = "planner"
    # STALE (2026-07-07): the route-to-END HITL pause is superseded by native
    # interrupt() inside the nodes — a paused run never reaches this router, so
    # the branch is dead. Kept per CLAUDE.md R13.
    # elif state.requires_approval and state.approval_token:
    #     dest = END  # paused for HITL (plan-stage gate or calendar-conflict approval)
    elif state.draft is not None and state.verdict is None:
        dest = "guardrails"
    else:
        dest = END
    log.debug("route_after_orchestrator", dest=dest, has_plan=state.plan is not None,
              has_draft=state.draft is not None, error=state.error)
    return dest  # type: ignore[return-value]


def _route_after_guard(state: AgentState) -> Literal["planner", "orchestrator", "__end__"]:
    """Map the verdict guardrails_node set onto a destination — pure, no mutation (R6)."""
    if state.error:
        dest: str = END
    elif state.verdict == "finalize":
        dest = "orchestrator"  # proceed to finalize/persist
    elif state.verdict == "replan":
        dest = "planner"
    else:  # "hitl", "block", or None — nothing left to run in the graph
        dest = END
    log.info("route_after_guard", dest=dest, verdict=state.verdict,
             confidence=round(state.confidence, 3), retry_count=state.retry_count)
    return dest  # type: ignore[return-value]


def _build_graph() -> StateGraph:
    """Assemble the StateGraph with all nodes and conditional edges."""
    graph = StateGraph(AgentState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("planner", planner_node)
    graph.add_node("guardrails", guardrails_node)
    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {"planner": "planner", "guardrails": "guardrails", END: END},
    )
    graph.add_edge("planner", "orchestrator")
    graph.add_conditional_edges(
        "guardrails",
        _route_after_guard,
        {"planner": "planner", "orchestrator": "orchestrator", END: END},
    )
    return graph


@lru_cache(maxsize=1)
def compile_graph():
    """Return the compiled LangGraph singleton bound to the active checkpointer.

    init_checkpointer() clears this cache after swapping in the Postgres saver,
    so anything compiled before the lifespan ran is recompiled on next use.
    """
    log.info("graph_compiled")
    return _build_graph().compile(checkpointer=_checkpointer)


async def discard_thread(trace_id: str) -> None:
    """Drop a TERMINAL run's checkpoints once its graph invocation has returned.

    Checkpoints are the HITL resume source now (native interrupt() + Command),
    so callers must only discard after a run truly finishes — never while it is
    paused awaiting approval. Best-effort: a failure here must never fail the run.
    # STALE (2026-07-07): previous rationale — "HITL resume rebuilds state from
    # Postgres rather than the checkpoint (this build routes to END and re-invokes
    # instead of using LangGraph interrupt())" — superseded by the native
    # interrupt() migration.
    """
    try:
        await compile_graph().checkpointer.adelete_thread(trace_id)
        log.debug("checkpoint_discarded", trace_id=trace_id)
    except Exception as exc:
        log.warning("checkpoint_discard_failed", trace_id=trace_id, error=str(exc))
