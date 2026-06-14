"""Guardrails node: deterministic checks, LLM judge, and confidence scoring."""

from __future__ import annotations

import re
from json import JSONDecodeError
from statistics import mean

import structlog
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from app.agents.response_composer import compose
from app.core.state import AgentState, DraftResponse, GuardVerdict
from app.data.repositories import create_approval, get_token, save_plan
from app.llm.client import get_structured_llm, run_metadata
from app.prompts import build_guard_judge_messages

log = structlog.get_logger(__name__)
# Guardrails confidence thresholds (also imported by the graph router). Kept here,
# with the guardrails node, so the scoring and routing bands share one definition.
_HIGH_CONFIDENCE = 0.85
_LOW_CONFIDENCE = 0.55
# When True, the medium-confidence band (0.55–0.85) pauses for HITL approval instead
# of auto-finalizing. Disabled: confidence is computed AFTER the tools run, so a medium
# score on a successful write would ask the user to approve an action that already
# happened (rejecting can't undo it). Approval is reserved for deliberate PRE-action
# gates instead — calendar clashes and destructive deletes, both handled in the
# orchestrator. Low confidence (<0.55) still re-plans/blocks via the graph router.
_HITL_ENABLED = False
# Above this judge-reported risk the draft is treated as possibly unsupported by
# the tool evidence and gets one corrective re-composition (see _mitigate_*).
_HALLUCINATION_RISK_LIMIT = 0.7
# Only fall back to a neutral verdict when the LLM responded with malformed
# output — transport errors (rate limit, timeout, API error) propagate so the
# run fails honestly instead of finalizing with a 0.5-judge confidence.
_SCHEMA_ERRORS = (ValidationError, JSONDecodeError, OutputParserException)

# Email addresses are intentionally NOT treated as blocking PII: this is an email
# assistant, so sender/recipient addresses are inherent to legitimate output.
# Only genuinely sensitive, low-false-positive patterns block the draft.
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),  # 16-digit credit-card number
]


def _contains_pii(text: str) -> bool:
    """Return True if any PII regex matches the draft markdown."""
    return any(p.search(text) for p in _PII_PATTERNS)


async def _stage_one(state: AgentState) -> bool:
    """Run deterministic checks. Returns True if the LLM judge should run next."""
    if state.draft and _contains_pii(state.draft.detail_markdown):
        log.warning("guard_blocked_pii", trace_id=state.trace_id)
        state.error = "pii_detected"
        state.confidence = 0.0
        state.requires_approval = True
        return False

    # Destructive actions are gated pre-execution in the orchestrator, not here — by the
    # time the guard runs the tools have already executed. This loop only verifies the
    # user still has a valid integration token for every tool the plan touches.
    if state.plan:
        for step in state.plan.steps:
            token = await get_token(state.user_id, step.tool)
            if not token:
                log.warning("guard_blocked_missing_integration", tool=step.tool)
                state.error = f"missing_integration:{step.tool}"
                state.confidence = 0.0
                return False
    log.debug("guard_stage_one_passed")
    return True


def _compute_confidence(state: AgentState, verdict: GuardVerdict) -> float:
    """Blend execution success, LLM-judge quality, and schema validity into a 0–1 score.

    Calibrated so a normal task NEVER needs human approval: when every tool call
    succeeded and the draft is schema-valid, the score is >= 0.9 — comfortably above
    the 0.85 finalize line — regardless of the judge. Execution success dominates
    (0.8); the judge and schema are minor adjusters (0.1 each). Retrieval similarity
    is deliberately excluded: a perfectly executed but novel request (no similar past
    plan) is still high-confidence — the old 0.3 similarity weight wrongly capped such
    runs at 0.7 and forced them into the medium band.

    Approval is therefore reserved for the deterministic PRE-action gates in the
    orchestrator — destructive deletes (always) and calendar clashes — not this score.
    Tool failures pull the score below 0.55 so the graph re-plans, then blocks.
    """
    if state.tool_results:
        tool_success_rate = sum(1 for r in state.tool_results if r.ok) / len(state.tool_results)
    else:
        tool_success_rate = 1.0

    llm_judge_avg = mean(
        [verdict.tone_fit, 1.0 - verdict.hallucination_risk, verdict.instruction_adherence]
    )

    schema_ok = 1.0 if isinstance(state.draft, DraftResponse) else 0.0

    return 0.8 * tool_success_rate + 0.1 * llm_judge_avg + 0.1 * schema_ok


async def _judge(state: AgentState) -> GuardVerdict:
    """Run the guard-judge LLM over the draft + tool-result evidence.

    Malformed judge JSON falls back to a neutral verdict so scoring proceeds from
    the deterministic signals; transport errors are NOT caught here and propagate.
    """
    messages = build_guard_judge_messages(
        draft=state.draft, user_request=state.user_request, tool_results=state.tool_results)
    llm = get_structured_llm("guard-judge", GuardVerdict, run_metadata(state))
    try:
        return await llm.ainvoke(messages)
    except _SCHEMA_ERRORS as exc:
        log.warning("guard_judge_fallback_neutral", error=str(exc))
        return GuardVerdict(tone_fit=0.5, hallucination_risk=0.5, instruction_adherence=0.5)


async def _mitigate_hallucination(state: AgentState, verdict: GuardVerdict) -> GuardVerdict:
    """Re-compose once against the evidence, then re-judge; warn if still risky.

    The tools have already executed, so re-planning would re-fire side effects.
    Instead the composer gets a single corrective pass constrained to the tool
    results, and the draft is re-judged. If it still reads as unsupported it
    ships with a visible warning and a ``degraded`` flag rather than silently.
    Returns the verdict that should drive the final confidence score.
    """
    log.warning("guard_hallucination_flagged", trace_id=state.trace_id,
                hallucination_risk=round(verdict.hallucination_risk, 3))
    feedback = (
        f"A previous draft was flagged for possible unsupported claims "
        f"(hallucination_risk={verdict.hallucination_risk:.2f}). Rewrite the response "
        f"using ONLY facts present in the tool results; remove or qualify any "
        f"statement not directly supported by them."
    )
    try:
        state.draft = await compose(state, judge_feedback=feedback)
    except Exception as exc:  # transport/compose failure — keep the original draft
        log.warning("guard_recompose_failed", trace_id=state.trace_id, error=str(exc))
        state.degraded.append("hallucination_recompose_failed")
        return verdict

    new_verdict = await _judge(state)
    if new_verdict.hallucination_risk > _HALLUCINATION_RISK_LIMIT and state.draft is not None:
        state.draft.detail_markdown = (
            "> ⚠️ **Heads-up:** parts of this response may not be fully supported by "
            "the data retrieved. Please verify before acting.\n\n"
        ) + state.draft.detail_markdown
        state.degraded.append("unverified_hallucination_risk")
        log.warning("guard_hallucination_unresolved", trace_id=state.trace_id,
                    hallucination_risk=round(new_verdict.hallucination_risk, 3))
    else:
        log.info("guard_hallucination_mitigated", trace_id=state.trace_id,
                 hallucination_risk=round(new_verdict.hallucination_risk, 3))
    return new_verdict


async def guardrails_node(state: AgentState) -> AgentState:
    """Score the draft, set confidence + requires_approval, and return updated state."""
    log.info("guardrails_node_start", trace_id=state.trace_id)
    should_continue = await _stage_one(state)
    if not should_continue or state.draft is None:
        log.info("guardrails_node_short_circuit", error=state.error)
        return state

    verdict = await _judge(state)
    # Teeth: a high hallucination_risk triggers one corrective re-composition
    # against the evidence (never a re-plan — tools already ran). The final
    # verdict (re-judged when mitigated) is what feeds the confidence score.
    if verdict.hallucination_risk > _HALLUCINATION_RISK_LIMIT:
        verdict = await _mitigate_hallucination(state, verdict)

    state.confidence = _compute_confidence(state, verdict)
    # Medium-confidence band pauses for HITL. Mint the approval token now so the sync
    # graph router can simply route to END and the UI has a valid token to act on.
    if (_HITL_ENABLED and _LOW_CONFIDENCE <= state.confidence < _HIGH_CONFIDENCE
            and not state.approval_token):
        state.requires_approval = True
        state.approval_token = await create_approval(state.trace_id)
        await save_plan(state)
        log.info("guard_awaiting_approval", trace_id=state.trace_id,
                 confidence=round(state.confidence, 3))
    log.info("guardrails_node_done", confidence=round(state.confidence, 3),
             requires_approval=state.requires_approval)
    return state
