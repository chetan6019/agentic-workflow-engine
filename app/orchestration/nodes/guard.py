"""Guardrails node: deterministic checks, LLM judge, and confidence scoring."""

from __future__ import annotations

import time
from json import JSONDecodeError
from statistics import mean

import structlog
from langchain_core.exceptions import OutputParserException
from opentelemetry import trace
from pydantic import ValidationError

from app.agents.response_composer import compose
# node_latency_seconds removed — node latency now comes from the OTel node span.
from app.core.metrics import guard_outcome_total
from app.core.state import AgentState, DraftResponse, GuardVerdict
from app.data.repositories import create_approval, get_token, save_plan
from app.guardrails import evaluate_output
from app.llm.client import get_structured_llm, run_metadata
from app.prompts import build_guard_judge_messages
from app.security.redaction import contains_pii

log = structlog.get_logger(__name__)
# No-op tracer unless app/core/tracing.py configured a provider at startup.
_tracer = trace.get_tracer(__name__)

# ============================================================
# CONFIGURATION (tweak these to change behavior)
# ============================================================
HIGH_CONFIDENCE = 0.85      # Auto-finalize above this
LOW_CONFIDENCE = 0.55       # Re-plan below this, block if retries exhausted
MAX_RETRIES = 2             # Max re-plans before blocking
HALLUCINATION_RISK_LIMIT = 0.85  # Trigger re-compose if judge risk exceeds this
HITL_ENABLED = False        # Medium-confidence pause for human approval (disabled)

# Exceptions that mean "judge gave bad JSON" — not "LLM is down"
SCHEMA_ERRORS = (ValidationError, JSONDecodeError, OutputParserException)

# ============================================================
# HELPER: PII check across all draft text fields
# ============================================================
def draft_has_pii(draft: DraftResponse) -> bool:
    """True if any user-facing field contains sensitive PII (SSN, credit card)."""
    # Email addresses are OK — this is an email assistant
    fields = [
        draft.summary,
        draft.detail_markdown,
        *draft.actions_taken,
        *draft.actions_pending,
    ]
    return any(contains_pii(text) for text in fields)

# ============================================================
# STAGE 1: Fast deterministic checks (no LLM calls)
# ============================================================
async def run_deterministic_checks(state: AgentState) -> bool:
    """
    Run quick checks. Return True if we should continue to LLM judge.
    Sets state.error / state.confidence / state.requires_approval on failure.
    """
    # 1. PII in draft?
    if state.draft and draft_has_pii(state.draft):
        log.warning("guard_blocked_pii", trace_id=state.trace_id)
        state.error = "pii_detected"
        state.confidence = 0.0
        state.requires_approval = True
        return False

    # 2. Missing integration tokens for any planned tool?
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

# ============================================================
# CONFIDENCE SCORING
# ============================================================
def compute_confidence(state: AgentState, verdict: GuardVerdict) -> float:
    """
    Blend three signals into 0–1 confidence:
      - 80% tool success rate
      - 10% LLM judge average (tone, factuality, adherence)
      - 10% schema validity
    """
    # Tool success rate (1.0 if no tools ran)
    if state.tool_results:
        tool_success = sum(1 for r in state.tool_results if r.ok) / len(state.tool_results)
    else:
        tool_success = 1.0

    # Judge average: tone_fit + (1 - hallucination_risk) + instruction_adherence
    judge_avg = mean([
        verdict.tone_fit,
        1.0 - verdict.hallucination_risk,
        verdict.instruction_adherence,
    ])

    # Schema validity
    schema_ok = 1.0 if isinstance(state.draft, DraftResponse) else 0.0

    return 0.8 * tool_success + 0.1 * judge_avg + 0.1 * schema_ok

# ============================================================
# LLM JUDGE: evaluates draft quality against evidence
# ============================================================
async def run_judge(state: AgentState) -> GuardVerdict:
    """Run the guard-judge LLM over the draft + tool-result evidence.
       Fallback to neutral verdict on malformed JSON only.
    """
    messages = build_guard_judge_messages(
        draft=state.draft, 
        user_request=state.user_request, 
        tool_results=state.tool_results
    )
    llm = get_structured_llm("guard-judge", GuardVerdict, run_metadata(state))

    try:
        return await llm.ainvoke(messages)
    except SCHEMA_ERRORS as exc:
        log.warning("guard_judge_fallback_neutral", error=str(exc))
        if "guard_judge_neutral_fallback" not in state.degraded:
            state.degraded.append("guard_judge_neutral_fallback")
        # Neutral middle-ground verdict
        return GuardVerdict(tone_fit=0.5, hallucination_risk=0.5, instruction_adherence=0.5)

# ============================================================
# HALLUCINATION MITIGATION: one corrective re-compose
# ============================================================
async def mitigate_hallucination(state: AgentState, verdict: GuardVerdict) -> GuardVerdict:
    """
    If judge flags high hallucination risk, re-compose once with strict
    "stick to evidence" instructions, then re-judge.
    """
    log.warning(
        "guard_hallucination_flagged",
        trace_id=state.trace_id,
        hallucination_risk=round(verdict.hallucination_risk, 3),
    )
    feedback = (
        f"Previous draft flagged for unsupported claims "
        f"(hallucination_risk={verdict.hallucination_risk:.2f}). "
        f"Rewrite using ONLY facts from tool results. "
        f"Remove or qualify anything not directly supported."
    )
    try:
        state.draft = await compose(state, judge_feedback=feedback)
    except Exception as exc:  # transport/compose failure — keep the original draft
        log.warning("guard_recompose_failed", trace_id=state.trace_id, error=str(exc))
        state.degraded.append("hallucination_recompose_failed")
        return verdict  # keep the original verdict

    # Re-judge the corrected draft
    new_verdict = await run_judge(state)

    if new_verdict.hallucination_risk > HALLUCINATION_RISK_LIMIT and state.draft is not None:
        # Still risky — add visible warning to response
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

# ============================================================
# VERDICT DECISION: map confidence → action
# ============================================================
def decide_verdict(state: AgentState) -> None:
    """
    Set state.verdict based on confidence bands.
    Pure logic — no I/O. Graph router reads state.verdict only.
    """
    if state.error:
        return  # Short-circuit: router will go to END
    conf = state.confidence

    if conf >= LOW_CONFIDENCE:
        if HITL_ENABLED and conf < HIGH_CONFIDENCE:
            state.verdict = "hitl"
            state.requires_approval = True
        else:
            state.verdict = "finalize"
            state.phase = "finalize"

    elif state.retry_count < MAX_RETRIES:
        # Re-plan: clear plan/results/draft for fresh attempt
        state.verdict = "replan"
        state.retry_count += 1
        state.plan = None
        state.tool_results = []
        state.draft = None
        state.confidence = 0.0

    else:
        state.verdict = "block"
        state.error = "low_confidence_blocked"

# ============================================================
# OUTPUT GUARDRAILS: final safety gate before streaming
# ============================================================
async def run_output_guardrails(state: AgentState) -> bool:
    """
    Apply output guardrails (PII redaction, toxicity, etc.).
    Returns True to continue; False to short-circuit (blocked or needs approval).
    """
    if state.draft is None:
        return True
    
    decision = await evaluate_output(
        state.draft.detail_markdown,
        draft=state.draft, plan=state.plan, retrieved_plans=state.retrieved_plans,
        user_id=state.user_id, trace_id=state.trace_id, source_was_user=False,
    )
    # Apply any redactions/truncations
    if decision.text != state.draft.detail_markdown:
        state.draft.detail_markdown = decision.text
    # Hard block
    if not decision.allowed:
        state.error = f"guardrail_output:{','.join(decision.blocked_rules)}"
        log.warning("output_guardrail_blocked", rules=decision.blocked_rules)
        return False
    # Needs human approval
    if decision.requires_approval and not state.approval_token:
        state.requires_approval = True
        state.approval_token = await create_approval(state.trace_id)
        await save_plan(state)  # persist token + redacted draft so resume finds them
        log.info("output_guardrail_awaiting_approval", trace_id=state.trace_id,
                 rules=[h.rule for h in decision.hits if h.action == "flag_for_approval"])
        return False
    return True

# ============================================================
# MAIN NODE: orchestrates the guardrails pipeline
# ============================================================
async def guardrails_node(state: AgentState) -> AgentState:
    """Main entry point — runs full guardrails pipeline.
       - Score the draft
       - set confidence + verdict
       - return updated state.
    """
    with structlog.contextvars.bound_contextvars(
        trace_id=state.trace_id,
        user_id=state.user_id,
        session_id=state.session_id,
        node="guardrails",
    ), _tracer.start_as_current_span("node.guardrails"):
        start = time.monotonic()
        log.info("guardrails_node_start", trace_id=state.trace_id)

        # Stage 1: Fast deterministic checks
        should_continue = await run_deterministic_checks(state)
        if not should_continue or state.draft is None:
            # Short-circuit before judge ever ran. Bucket as "block" because the
            # deterministic check set state.error and won't proceed — same effect
            # as a low-confidence block from the policy point of view.
            guard_outcome_total.labels(outcome="block").inc()
            log.info("guardrails_node_short_circuit", error=state.error)
            return state
        
        # Stage 2: LLM judge evaluates draft quality
        verdict = await run_judge(state)

        # Stage 3: Hallucination mitigation (re-compose once if needed)
        if verdict.hallucination_risk > HALLUCINATION_RISK_LIMIT:
            verdict = await mitigate_hallucination(state, verdict)

        # Stage 4: Output guardrails(final safety gate)(redact/flag/block) before it can finalize.
        if not await run_output_guardrails(state):
            # Output gate failed — either hard block or "requires approval".
            # `state.requires_approval` distinguishes the two; map to existing
            # verdict vocabulary so the metric uses one consistent dialect.
            guard_outcome_total.labels(
                outcome="hitl" if state.requires_approval else "block"
            ).inc()
            log.info("guardrails_node_output_gate", error=state.error,
                    requires_approval=state.requires_approval)
            return state

        # Stage 5: Compute confidence & decide verdict
        state.confidence = compute_confidence(state, verdict)
        decide_verdict(state)

        # Record the verdict outcome for the "auto-approval rate" metric.
        # decide_verdict can leave state.verdict as None if state.error short-
        # circuits — use "block" as the fallback so every guard call contributes
        # exactly one observation (counters with missing samples skew rates).
        guard_outcome_total.labels(outcome=state.verdict or "block").inc()

        # Mint approval token if HITL (inert while HITL_ENABLED=False)
        if state.verdict == "hitl" and not state.approval_token:
            state.approval_token = await create_approval(state.trace_id)
            await save_plan(state)
            log.info("guard_awaiting_approval", trace_id=state.trace_id,
                    confidence=round(state.confidence, 3))
        
        # node_latency_seconds observation removed — the OTel node span times this.
        duration_ms = int((time.monotonic() - start) * 1000)
        log.info("guardrails_node_done",
                confidence=round(state.confidence, 3),
                verdict=state.verdict, 
                requires_approval=state.requires_approval,
                duration_ms=duration_ms)
        
        return state
