"""LangGraph wiring: StateGraph nodes, conditional edges, and compiled singleton."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.core.state import AgentState
from app.orchestration.nodes.guard import (
    _HIGH_CONFIDENCE,
    _HITL_ENABLED,
    _LOW_CONFIDENCE,
    guardrails_node,
)
from app.orchestration.nodes.orchestrator import orchestrator_node
from app.orchestration.nodes.planner import planner_node

log = structlog.get_logger(__name__)

_MAX_RETRIES = 2


def _route_after_orchestrator(state: AgentState) -> Literal["planner", "guardrails", "__end__"]:
    """Route to planner before a plan exists, to guardrails once a draft exists."""
    if state.error:
        dest: str = END
    elif state.plan is None:
        dest = "planner"
    elif state.requires_approval and state.approval_token:
        dest = END  # paused for HITL (plan-stage gate or calendar-conflict approval)
    elif state.draft is not None and state.confidence == 0.0:
        dest = "guardrails"
    else:
        dest = END
    log.debug("route_after_orchestrator", dest=dest, has_plan=state.plan is not None,
              has_draft=state.draft is not None, error=state.error)
    return dest  # type: ignore[return-value]


def _route_after_guard(state: AgentState) -> Literal["planner", "orchestrator", "__end__"]:
    """Finalize, pause for HITL, retry, or block — based on guard confidence."""
    conf = state.confidence
    if state.error:
        dest: str = END
    elif conf >= _LOW_CONFIDENCE:
        # >= 0.85 always finalizes. The 0.55–0.85 band finalizes too while HITL
        # is idle; flip _HITL_ENABLED to pause it for approval instead.
        if _HITL_ENABLED and conf < _HIGH_CONFIDENCE:
            state.requires_approval = True  # UI shows HITL; approval re-invokes the graph
            dest = END
        else:
            dest = "orchestrator"  # proceed to finalize/persist
    elif state.retry_count < _MAX_RETRIES:
        state.retry_count += 1
        state.plan = None
        state.tool_results = []
        state.draft = None
        state.confidence = 0.0
        dest = "planner"
    else:
        state.error = "low_confidence_blocked"
        dest = END
    log.info("route_after_guard", dest=dest, confidence=round(conf, 3),
             retry_count=state.retry_count, requires_approval=state.requires_approval)
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
