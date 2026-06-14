"""Planner node: turns the user request into a typed ExecutionPlan."""

from __future__ import annotations

import datetime
import time
from json import JSONDecodeError
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from app.config import get_settings
from app.core.state import AgentState, ExecutionPlan, ToolSpec
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


def _invalid_steps(plan: ExecutionPlan, tool_specs: list[ToolSpec]) -> list[str]:
    """Return descriptions of steps whose tool/action aren't in the catalog.

    The planner contract is tool=server, action=tool-name; the swapped form is
    accepted too because the MCP client resolves it. Empty list = every step
    maps to a known tool, so we catch hallucinated names here — before any tool
    runs — instead of as a runtime unknown_tool failure mid-execution.
    """
    valid = {(s.server, s.name) for s in tool_specs}
    return [f"{step.id}:{step.tool}/{step.action}" for step in plan.steps
            if (step.tool, step.action) not in valid
            and (step.action, step.tool) not in valid]


async def planner_node(state: AgentState) -> AgentState:
    """Build an ExecutionPlan from the user request, retrieved plans, and tool specs."""
    t0 = time.monotonic()
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
        history=state.history,
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

    # Validate the plan against the known tools before anything executes. Skip when
    # the catalog is empty (unreachable) — don't fail a run on an infra hiccup.
    if tool_specs and (bad := _invalid_steps(plan, tool_specs)):
        log.warning("planner_plan_invalid", steps=bad)
        correction = HumanMessage(content=(
            "<correction>\nThe previous plan referenced tools/actions not in "
            f"<tools>: {'; '.join(bad)}. Re-output the FULL plan using ONLY the "
            "listed tools, with `tool` = the server and `action` = the tool name."
            "\n</correction>"))
        try:
            plan = await _invoke_planner("planner-default", [*messages, correction], meta)
        except Exception as exc:
            log.error("planner_revalidation_failed", error=str(exc))
            state.error = f"plan_validation_failed: {exc!s}"
            return state
        if still_bad := _invalid_steps(plan, tool_specs):
            log.error("planner_plan_still_invalid", steps=still_bad)
            state.error = f"plan_validation_failed:{','.join(still_bad)}"
            return state

    state.plan = plan
    # A fresh plan advances the explicit phase to execute and clears any verdict left
    # over from a re-plan, so the next orchestrator pass executes and the post-execute
    # router (verdict is None) sends the draft back to guardrails.
    state.phase = "execute"
    state.verdict = None
    log.info("planner_node_done", steps=len(plan.steps), strategy=plan.strategy,
             complexity=plan.complexity_score,
             duration_ms=int((time.monotonic() - t0) * 1000))
    return state
