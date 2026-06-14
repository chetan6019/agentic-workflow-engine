"""LangGraph wiring: StateGraph nodes, conditional edges, and compiled singleton."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.core.state import AgentState
from app.orchestration.nodes.guard import guardrails_node
from app.orchestration.nodes.orchestrator import orchestrator_node
from app.orchestration.nodes.planner import planner_node

log = structlog.get_logger(__name__)


def _route_after_orchestrator(state: AgentState) -> Literal["planner", "guardrails", "__end__"]:
    """Route to planner before a plan exists, to guardrails once an unscored draft exists.

    Pure: reads state only. ``verdict is None`` means guardrails hasn't run yet, so a
    fresh draft is sent there to be scored; once a verdict exists the run is done.
    """
    if state.error:
        dest: str = END
    elif state.plan is None:
        dest = "planner"
    elif state.requires_approval and state.approval_token:
        dest = END  # paused for HITL (plan-stage gate or calendar-conflict approval)
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
    """Return the compiled LangGraph singleton with an in-memory checkpointer."""
    log.info("graph_compiled")
    return _build_graph().compile(checkpointer=MemorySaver())


async def discard_thread(trace_id: str) -> None:
    """Drop a run's in-memory checkpoints once its graph invocation has returned.

    HITL resume rebuilds state from Postgres rather than the checkpoint (this build
    routes to END and re-invokes instead of using LangGraph interrupt()), so a
    thread's checkpoints are pure memory growth once the invocation completes.
    Best-effort: a failure here must never fail the run.
    """
    try:
        await compile_graph().checkpointer.adelete_thread(trace_id)
        log.debug("checkpoint_discarded", trace_id=trace_id)
    except Exception as exc:
        log.warning("checkpoint_discard_failed", trace_id=trace_id, error=str(exc))
