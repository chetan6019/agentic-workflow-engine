"""Planner node: turn a user request into a typed ExecutionPlan.

The planner is the LangGraph node that reads the user's request, retrieved
past plans, and the tool catalog, then asks the LLM for a JSON plan we can
validate against `app.core.state.ExecutionPlan`. Before handing the plan to
the executor, this module also checks every step references a tool that
actually exists — catching hallucinated tool names here is much cheaper
than discovering them mid-execution.

Public surface (do not rename without updating callers):
    - planner_node(state) -> AgentState       : the LangGraph node entry point.
    - generate_plan_with_retry(...)           : LLM call + one schema-shaped retry.
    - validate_and_fix_plan(...)              : tool-catalog validation + one re-ask.
    - call_planner_llm(...)                   : a single structured-output LLM call.
    - find_invalid_plan_steps(...)            : pure validation primitive.
    - build_correction_message(...)           : re-ask prompt builder.
    - build_weekday_reference(...)            : weekday/date table for the prompt.
"""

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
from app.core.metrics import node_latency_seconds, planner_outcome_total
from app.core.state import AgentState, ExecutionPlan, ToolSpec
from app.llm.client import call_metadata, get_structured_llm
from app.prompts import build_planner_messages
from app.rag.tool_docs import fetch_tool_specs

log = structlog.get_logger(__name__)

# How many tool docs to fetch from the RAG layer for the planner prompt.
# Small enough that the prompt stays compact across the four MCP servers,
# large enough that the planner usually sees the tools it actually needs.
TOOL_DOC_LIMIT = 4

# LiteLLM router alias used for every planner LLM call. Hoisted so it isn't
# repeated as a magic string in three places.
PLANNER_ROLE = "planner-default"

# The error shapes that mean "the model returned something we couldn't turn
# into an ExecutionPlan". A single retry against these is worth the cost —
# models occasionally lose JSON formatting under load — whereas any OTHER
# exception (network, auth, ...) won't be fixed by retrying.
SCHEMA_ERRORS = (ValidationError, JSONDecodeError, OutputParserException)


def build_weekday_reference(now: datetime.datetime) -> str:
    """Return a literal "weekday => YYYY-MM-DD" table for the next 7 days.

    LLMs are unreliable at computing future dates ("what date is next Monday?").
    Handing them an explicit lookup table removes the whole class of weekday-
    math errors when the user says things like "this Friday" or "coming Tuesday".
    """
    today = now.date()
    lines: list[str] = [f"Today is {today.strftime('%A')} {today.isoformat()}."]
    for offset in range(1, 8):
        future_date = today + datetime.timedelta(days=offset)
        lines.append(f"{future_date.strftime('%A')} = {future_date.isoformat()}")
    return "\n".join(lines)


async def call_planner_llm(
    role: str,
    messages: list[Any],
    metadata: dict[str, str],
) -> ExecutionPlan:
    """Make one structured-output planner LLM call and return the parsed plan.

    `with_structured_output` (inside get_structured_llm) parses the model's
    JSON into an ExecutionPlan for us. Exceptions are intentionally NOT caught
    here — the caller decides whether the failure is retryable (see
    generate_plan_with_retry and validate_and_fix_plan).

    `role` is kept as a parameter for symmetry with get_structured_llm even
    though every call site passes PLANNER_ROLE today.
    """
    llm = get_structured_llm(role, ExecutionPlan, metadata)
    return await llm.ainvoke(messages)


def find_invalid_plan_steps(
    plan: ExecutionPlan,
    tool_specs: list[ToolSpec],
) -> list[str]:
    """Return one descriptor per plan step that references a tool we don't have.

    The planner contract is `tool` = MCP server, `action` = tool name. We also
    accept the SWAPPED form (tool/action transposed) because the MCP client
    transparently resolves it — but a fully invented name still has to be
    caught here, before the executor calls into a nonexistent tool.

    An empty result means every step is well-formed. Each invalid step is
    rendered as "<step_id>:<tool>/<action>" so the correction message can
    name them specifically.
    """
    known_pairs: set[tuple[str, str]] = {
        (tool.server, tool.name) for tool in tool_specs
    }

    invalid_steps: list[str] = []
    for step in plan.steps:
        # Two acceptable orderings: as the planner emitted, or with tool/action
        # transposed. Anything else is a hallucinated name.
        as_emitted = (step.tool, step.action)
        as_swapped = (step.action, step.tool)
        if as_emitted not in known_pairs and as_swapped not in known_pairs:
            invalid_steps.append(f"{step.id}:{step.tool}/{step.action}")

    return invalid_steps


def build_correction_message(invalid_steps: list[str]) -> HumanMessage:
    """Build a user-turn correction message naming the invalid steps explicitly.

    Used after find_invalid_plan_steps returns hits — appended to the original
    planner messages as a second user turn so the model sees both its own
    bad output (already in the history) and what to fix.
    """
    invalid_text = "; ".join(invalid_steps)
    content = (
        "The previous plan referenced tools/actions not in the tool list: "
        f"{invalid_text}. Re-output the FULL plan using ONLY the listed tools, "
        "with `tool` = the server and `action` = the tool name."
    )
    return HumanMessage(content=content)


async def generate_plan_with_retry(
    state: AgentState,
    messages: list[Any],
    metadata: dict[str, str],
) -> ExecutionPlan:
    """Call the planner once. On a schema-shaped failure, retry exactly once.

    Two attempts, not more — schema errors usually clear on a second sample,
    and chaining more retries hides genuine prompt problems while burning
    cost. The retry feeds the parse error back to the model as an additional
    user turn so it has signal about what to fix; without that, retries are
    just hopeful re-rolls.

    Any non-schema exception is wrapped in RuntimeError so the planner_node
    error path (which only catches RuntimeError) can react uniformly.
    """
    # ---- First attempt: the happy path. -----------------------------------
    try:
        return await call_planner_llm(PLANNER_ROLE, messages, metadata)
    except SCHEMA_ERRORS as exc:
        # Recoverable shape error — fall through to the retry below.
        log.warning("planner_schema_retry", error=str(exc))
        first_error = exc
    except Exception as exc:
        # Non-schema failure (network, auth, ...). Don't retry; wrap and raise.
        log.error("planner_failed", error=str(exc))
        raise RuntimeError(f"planner_failed: {exc!s}") from exc

    # ---- Second attempt: only reached after a SCHEMA_ERRORS member. -------
    # Append a correction turn that names the validation error explicitly,
    # and recompute call metadata so the retry gets its own prompt fingerprint
    # in Langfuse (same trace_id, distinct fingerprint = retry is identifiable).
    retry_msg = HumanMessage(
        content=(
            "The previous output failed schema validation: "
            f"{type(first_error).__name__}: {first_error!s}. "
            "Re-emit the FULL plan as valid JSON matching the ExecutionPlan schema."
        )
    )
    retry_messages = [*messages, retry_msg]
    retry_metadata = call_metadata(state, retry_messages)

    try:
        return await call_planner_llm(PLANNER_ROLE, retry_messages, retry_metadata)
    except Exception as retry_exc:
        log.error("planner_failed", error=str(retry_exc))
        raise RuntimeError(
            f"planner_structured_output_failed: {retry_exc!s}"
        ) from retry_exc


async def validate_and_fix_plan(
    state: AgentState,
    plan: ExecutionPlan,
    tool_specs: list[ToolSpec],
    original_messages: list[Any],
    metadata: dict[str, str],
) -> ExecutionPlan:
    """Check the plan uses real tools; if not, ask the LLM to regenerate once.

    Sequence:
      1. Identify any plan steps whose (tool, action) isn't in tool_specs.
      2. If none, return the plan unchanged.
      3. Otherwise, append a correction message and re-call the planner.
      4. Re-validate the corrected plan; if it's still bad, raise.

    Single-shot correction matches CLAUDE.md R12's "max 2 iterations"
    discipline — one initial plan, one corrected plan, then we stop trying.
    """
    # No tool catalog means there's nothing to check against. Trust the plan.
    if not tool_specs:
        return plan

    invalid_steps = find_invalid_plan_steps(plan, tool_specs)
    if not invalid_steps:
        return plan

    log.warning("planner_plan_invalid", steps=invalid_steps)

    # Re-ask the LLM with the original messages plus a correction turn that
    # names the bad steps. We recompute call_metadata so the corrected call
    # gets its own prompt fingerprint (same trace_id, distinct fingerprint).
    correction = build_correction_message(invalid_steps)
    corrected_messages = [*original_messages, correction]
    correction_metadata = call_metadata(state, corrected_messages)

    try:
        corrected_plan = await call_planner_llm(
            PLANNER_ROLE, corrected_messages, correction_metadata
        )
    except Exception as exc:
        log.error("planner_revalidation_failed", error=str(exc))
        raise RuntimeError(f"plan_validation_failed: {exc!s}") from exc

    still_invalid = find_invalid_plan_steps(corrected_plan, tool_specs)
    if still_invalid:
        # We tried once. Two failures in a row means a deeper prompt or
        # catalog issue — surface it as a hard error instead of looping.
        log.error("planner_plan_still_invalid", steps=still_invalid)
        invalid_text = ",".join(still_invalid)
        raise RuntimeError(f"plan_validation_failed:{invalid_text}")

    return corrected_plan


async def planner_node(state: AgentState) -> AgentState:
    """LangGraph node: request → planner messages → ExecutionPlan → state.

    Sequence:
      1. Bind structlog context so every log line in this node carries the
         trace/session/user ids (contextvar bindings propagate into any
         tasks spawned underneath).
      2. Build the planner prompt: tool catalog, current time + timezone,
         and the weekday reference table.
      3. Ask the LLM for a structured plan (with one schema retry).
      4. Validate the plan only references known tools (with one correction).
      5. Stash the plan on state and advance the phase to "execute".

    On RuntimeError we record state.error, set state.phase = "error", and
    return — the graph routes to the error handler from there. Other
    exception types are deliberately NOT caught; we want truly unexpected
    failures to propagate with their original traceback intact.

    Latency is observed in a `finally` block so failed runs still contribute
    to the histogram (otherwise p95 looks artificially good when the planner
    is failing).
    """
    with structlog.contextvars.bound_contextvars(
        trace_id=state.trace_id,
        user_id=state.user_id,
        session_id=state.session_id,
        node="planner",
    ):
        start_time = time.monotonic()
        log.info(
            "planner_node_start",
            retry_count=state.retry_count,
            examples=len(state.retrieved_plans),
        )

        try:
            # Step 2: gather everything the prompt needs.
            tool_specs = await fetch_tool_specs(state.user_request, TOOL_DOC_LIMIT)
            timezone_name = get_settings().default_tz
            now_local = datetime.datetime.now(ZoneInfo(timezone_name))

            messages = build_planner_messages(
                user_request=state.user_request,
                examples=state.retrieved_plans,
                tool_specs=tool_specs,
                now_local=now_local.isoformat(),
                tz_name=timezone_name,
                weekday_ref=build_weekday_reference(now_local),
                history=state.history,
            )
            metadata = call_metadata(state, messages)

            # Steps 3 + 4: generate, then validate (each with one retry).
            plan = await generate_plan_with_retry(state, messages, metadata)
            # Sample first-try validity BEFORE validate_and_fix_plan runs, so
            # we can tell the planner_outcome counter whether the model got it
            # right on attempt 1 or needed self-correction. This is cheap —
            # the same check runs again inside validate_and_fix_plan.
            first_try_invalid = bool(find_invalid_plan_steps(plan, tool_specs))
            plan = await validate_and_fix_plan(
                state=state,
                plan=plan,
                tool_specs=tool_specs,
                original_messages=messages,
                metadata=metadata,
            )

            # Step 5: commit the plan + advance the phase. `verdict` is
            # cleared so a stale guard verdict from an earlier run can't
            # leak through into the next phase.
            state.plan = plan
            state.phase = "execute"
            state.verdict = None

            # Record the outcome for the "plan validity" metric.
            planner_outcome_total.labels(
                outcome="self_corrected" if first_try_invalid else "valid"
            ).inc()

            duration_ms = int((time.monotonic() - start_time) * 1000)
            log.info(
                "planner_node_done",
                steps=len(plan.steps),
                strategy=plan.strategy,
                complexity=plan.complexity_score,
                first_try_invalid=first_try_invalid,
                duration_ms=duration_ms,
            )
        except RuntimeError as exc:
            # Expected failure mode — record and let the graph route to the
            # error handler. The traceback is preserved via `raise ... from`
            # at the original raise site.
            state.error = str(exc)
            state.phase = "error"
            planner_outcome_total.labels(outcome="failed").inc()
            log.error("planner_node_failed", error=str(exc))
        finally:
            # Observe latency on BOTH success and failure paths so error
            # latencies aren't silently excluded from the histogram.
            duration = time.monotonic() - start_time
            node_latency_seconds.labels(node="planner").observe(duration)

        return state
