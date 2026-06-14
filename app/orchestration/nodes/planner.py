"""Planner node: turns the user request into a typed ExecutionPlan."""

from __future__ import annotations

import datetime
from json import JSONDecodeError
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from app.config import get_settings
from app.core.state import AgentState, ExecutionPlan
from app.llm.client import get_structured_llm, run_metadata
from app.prompts import build_planner_messages
from app.rag.tool_docs import fetch_tool_specs

log = structlog.get_logger(__name__)
_TOOL_DOC_K = 6
# Schema/JSON failures = the LLM responded but the body was malformed — a
# *successful* HTTP call the LiteLLM gateway will not retry, so retry once
# locally. Transport errors (rate limit, timeout, network) are already retried
# by the gateway (num_retries) and its cross-provider fallbacks; when those are
# exhausted the run fails honestly. Same split as guard.py / response_composer.py.
_SCHEMA_ERRORS = (ValidationError, JSONDecodeError, OutputParserException)


def _weekday_reference(now: datetime.datetime) -> str:
    """Precompute the next future date for each weekday so the LLM never does date math.

    "coming Monday" / "this Friday" resolve to the nearest strictly-future date with
    that weekday name. Grounding the planner with exact dates removes the single most
    common planning error (wrong weekday).
    """
    today = now.date()
    lines = []
    for offset in range(1, 8):
        d = today + datetime.timedelta(days=offset)
        lines.append(f"{d.strftime('%A')} = {d.isoformat()}")
    return f"Today is {today.strftime('%A')} {today.isoformat()}.\n" + "\n".join(lines)


async def _invoke_planner(role: str, messages: list[Any],
                          metadata: dict[str, str]) -> ExecutionPlan:
    """Call the planner LLM with structured output and return an ExecutionPlan."""
    llm = get_structured_llm(role, ExecutionPlan, metadata)
    return await llm.ainvoke(messages)


async def planner_node(state: AgentState) -> AgentState:
    """Build an ExecutionPlan from the user request, retrieved plans, and tool specs."""
    log.info("planner_node_start", retry_count=state.retry_count,
             examples=len(state.retrieved_plans))
    tool_specs = await fetch_tool_specs(state.user_request, _TOOL_DOC_K)
    tz = get_settings().default_tz
    now = datetime.datetime.now(ZoneInfo(tz))
    messages = build_planner_messages(
        user_request=state.user_request,
        examples=state.retrieved_plans,
        tool_specs=tool_specs,
        now_local=now.isoformat(),
        tz_name=tz,
        weekday_ref=_weekday_reference(now),
    )

    meta = run_metadata(state)
    try:
        plan = await _invoke_planner("planner-default", messages, meta)
    except _SCHEMA_ERRORS as schema_exc:
        log.warning("planner_schema_retry", error=str(schema_exc))
        try:
            plan = await _invoke_planner("planner-default", messages, meta)
        except Exception as exc:
            log.error("planner_failed", error=str(exc))
            state.error = f"planner_structured_output_failed: {exc!s}"
            return state
    except Exception as exc:
        log.error("planner_failed", error=str(exc))
        state.error = f"planner_failed: {exc!s}"
        return state

    state.plan = plan
    log.info("planner_node_done", steps=len(plan.steps), strategy=plan.strategy,
             complexity=plan.complexity_score)
    return state
