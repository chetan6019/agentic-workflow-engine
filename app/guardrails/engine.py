"""Guardrail engine: runs the rule lists, folds Decisions, emits observability.

Two entrypoints — ``evaluate_input`` (pre-planner) and ``evaluate_output``
(pre-finalize). Redis-backed caps are fail-open: a Redis error never blocks a
caller. Every hit emits a structlog event; blocks/flags/rate-limits increment a
Prometheus counter and (when configured) a Langfuse event.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import structlog
import yaml

from app.config import get_settings
from app.core.metrics import guardrail_block_total
from app.core.state import DraftResponse, ExecutionPlan, RetrievedPlan
from app.data.redis_client import get_redis
from app.data.repositories import save_audit
from app.guardrails.contracts import Decision, RuleHit, decide
from app.guardrails.input_rules import (
    check_jailbreak,
    # STALE (2026-06-22): check_language retired (low-value signal); no longer imported.
    # check_language,
    check_oversized,
    check_pii,
    check_prompt_injection,
    check_secrets,
)
from app.guardrails.output_rules import (
    check_destructive,
    check_hallucination_citation,
    check_length,
    check_pii_leak,
    check_policy_tool_args,
    check_tool_args,
    # STALE (2026-06-22): check_profanity retired (subjective, low value); no longer imported.
    # check_profanity,
    check_secret_leak,
    is_trusted_read,
)

log = structlog.get_logger(__name__)
_POLICY_PATH = Path(__file__).with_name("policy.yaml")


class _Policy(NamedTuple):
    """Parsed view of policy.yaml (cached)."""

    denied: set[tuple[str, str]]
    approval_actions: set[str]
    email_action: str
    recipient_allowlist: set[str]
    profanity: list[str]
    max_emails_per_hour: int
    rate_per_minute: int
    rate_per_hour: int
    # Tool-specific config consumed by output_rules.check_tool_args + the trusted-read carve-out.
    google_event_cfg: dict[str, object]
    github_body_cfg: dict[str, object]
    finnhub_cfg: dict[str, object]
    trusted_read_actions: set[str]


@lru_cache(maxsize=1)
def _policy() -> _Policy:
    """Load + cache policy.yaml into a typed structure."""
    raw = yaml.safe_load(_POLICY_PATH.read_text(encoding="utf-8")) or {}
    denied = {(d["tool"], d["action"]) for d in (raw.get("denied_tool_actions") or [])}
    return _Policy(
        denied=denied,
        approval_actions=set(raw.get("approval_required_actions") or []),
        email_action=str(raw.get("email_send_action", "gmail.send_email")),
        recipient_allowlist=set(raw.get("recipient_allowlist") or []),
        profanity=list(raw.get("profanity") or []),
        max_emails_per_hour=int(raw.get("max_emails_per_hour", 5)),
        rate_per_minute=int(raw.get("rate_per_minute", 30)),
        rate_per_hour=int(raw.get("rate_per_hour", 300)),
        google_event_cfg=dict(raw.get("google_create_event") or {}),
        github_body_cfg=dict(raw.get("github_write_body") or {}),
        finnhub_cfg=dict(raw.get("finnhub") or {}),
        trusted_read_actions=set(raw.get("trusted_read_actions") or []),
    )


async def _incr(key: str, amount: int, ttl: int) -> int:
    """INCRBY a counter (TTL on first write); fail-open to 0 on any Redis error."""
    try:
        redis = get_redis()
        value = int(await redis.incrby(key, amount))
        if value == amount:
            await redis.expire(key, ttl)
        return value
    except Exception as exc:  # fail-open: a cap must never lock callers out
        log.warning("guardrail_redis_failed", key=key, error=str(exc))
        return 0


async def _rate_cap_hit(user_id: str) -> RuleHit | None:
    """Per-user per-minute / per-hour request caps via Redis counters."""
    pol = _policy()
    now = time.time()
    minute = await _incr(f"gr:rate:min:{user_id}:{int(now // 60)}", 1, 60)
    hour = await _incr(f"gr:rate:hr:{user_id}:{int(now // 3600)}", 1, 3600)
    if minute > pol.rate_per_minute or hour > pol.rate_per_hour:
        return RuleHit(rule="rate_cap", action="rate_limit", severity="medium",
                       detail="per-user request rate cap exceeded")
    return None


async def _mass_email_hit(plan: ExecutionPlan | None, user_id: str) -> RuleHit | None:
    """Block when outbound emails this hour exceed the configured ceiling."""
    pol = _policy()
    if plan is None:
        return None
    sends = sum(1 for s in plan.steps if f"{s.tool}.{s.action}" == pol.email_action)
    if sends == 0:
        return None
    total = await _incr(f"gr:mass_email:{user_id}:{int(time.time() // 3600)}", sends, 3600)
    if total > pol.max_emails_per_hour:
        return RuleHit(rule="mass_action_throttle", action="block", severity="high",
                       detail=f"outbound email throttle exceeded ({total}/{pol.max_emails_per_hour} per hour)")
    return None


async def _audit_secret_leak(user_id: str, trace_id: str) -> None:
    """Record a secret-leak block to structlog + AuditLog (best-effort, value-free)."""
    log.warning("guardrail_secret_leak_audit", trace_id=trace_id, user_id=user_id)
    try:
        await save_audit(user_id=user_id, action="guardrail_secret_leak", detail=f"trace={trace_id}")
    except Exception as exc:
        log.warning("guardrail_audit_persist_failed", trace_id=trace_id, error=str(exc))


@lru_cache(maxsize=1)
def _langfuse_client() -> object | None:
    """Lazily build a Langfuse client; None if the SDK or config is unavailable."""
    try:
        from langfuse import Langfuse

        return Langfuse()
    except Exception:
        return None


def _maybe_langfuse(direction: str, trace_id: str, hits: tuple[RuleHit, ...]) -> None:
    """Emit a Langfuse event per evaluation when keys are configured (best-effort)."""
    settings = get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key) or not hits:
        return
    client = _langfuse_client()
    if client is None:
        return
    try:
        client.create_event(  # type: ignore[attr-defined]
            name=f"guardrail_{direction}",
            metadata={"trace_id": trace_id,
                      "rules": [h.rule for h in hits],
                      "actions": [h.action for h in hits]},
        )
    except Exception as exc:
        log.debug("guardrail_langfuse_failed", error=str(exc))


def _emit(decision: Decision, trace_id: str) -> None:
    """Structlog every hit; count enforcement actions; mirror to Langfuse."""
    for hit in decision.hits:
        log.info("guardrail_hit", rule=hit.rule, action=hit.action, severity=hit.severity,
                 direction=decision.direction, trace_id=trace_id, detail=hit.detail)
        if hit.action in ("block", "flag_for_approval", "rate_limit"):
            guardrail_block_total.labels(rule=hit.rule, direction=decision.direction).inc()
    _maybe_langfuse(decision.direction, trace_id, decision.hits)


async def evaluate_input(text: str, *, user_id: str, trace_id: str) -> Decision:
    """Run the INPUT pipeline over a user request; returns a folded Decision."""
    hits: list[RuleHit] = []
    # STALE (2026-06-22): check_language removed from this tuple (retired, low-value signal).
    for check in (check_prompt_injection, check_jailbreak, check_secrets,
                  check_oversized):  # was: ..., check_oversized, check_language
        hit = check(text)
        if hit is not None:
            hits.append(hit)
    text, pii = check_pii(text)
    if pii is not None:
        hits.append(pii)
    rate = await _rate_cap_hit(user_id)
    if rate is not None:
        hits.append(rate)
    decision = decide("in", text, hits)
    _emit(decision, trace_id)
    return decision


async def evaluate_output(text: str, *, draft: DraftResponse | None,
                          plan: ExecutionPlan | None, retrieved_plans: list[RetrievedPlan],
                          user_id: str, trace_id: str, source_was_user: bool) -> Decision:
    """Run the OUTPUT pipeline over a composed response; returns a folded Decision."""
    pol = _policy()
    hits: list[RuleHit] = []

    # Trusted-read carve-out (CLAUDE.md rule #6): when the plan is non-empty and EVERY step is a
    # trusted read (the user's own github repos, or finnhub's structured JSON), skip the three
    # response content scans below — secret leak, PII leak, hallucination citation. The source is
    # trusted by definition, so scanning every response costs more than the threat. Per-user auth
    # and the Redis rate caps still run for every step; only these scans are skipped.
    trusted = plan is not None and bool(plan.steps) and all(
        is_trusted_read(step, pol.trusted_read_actions) for step in plan.steps)
    if trusted and plan is not None:
        log.info("guardrail_trusted_read_skip", trace_id=trace_id,
                 steps=[f"{s.tool}.{s.action}" for s in plan.steps])

    if not trusted:
        secret = check_secret_leak(text)
        if secret is not None:
            hits.append(secret)
            await _audit_secret_leak(user_id, trace_id)

        # PII gates approval only when the response is outbound (an email send leaves the
        # user's trust boundary). Reading the user's own data back to them just redacts.
        outbound = plan is not None and any(
            f"{s.tool}.{s.action}" == pol.email_action for s in plan.steps)
        text, pii = check_pii_leak(text, source_was_user=source_was_user, outbound=outbound)
        if pii is not None:
            hits.append(pii)

    # STALE (2026-06-22): check_profanity retired (subjective, low value); no longer called.
    # text, profanity = check_profanity(text, pol.profanity)
    # if profanity is not None:
    #     hits.append(profanity)

    # Policy / tool-arg / destructive checks run for EVERY plan — a trusted read is NOT exempt
    # from these (a denied action or oversized write arg must still be caught). check_tool_args
    # slots between the deny-list check and the destructive-action check per the migration spec.
    for hit in (check_policy_tool_args(plan, pol.denied),
                check_tool_args(plan, pol.google_event_cfg, pol.github_body_cfg),
                check_destructive(plan, approval_actions=pol.approval_actions,
                                  email_action=pol.email_action,
                                  recipient_allowlist=pol.recipient_allowlist),
                await _mass_email_hit(plan, user_id)):
        if hit is not None:
            hits.append(hit)

    # Hallucination-citation is a content scan on the draft, so it too is skipped for trusted reads.
    if not trusted:
        hallucination = check_hallucination_citation(draft, {p.plan_id for p in retrieved_plans})
        if hallucination is not None:
            hits.append(hallucination)

    text, length = check_length(text)
    if length is not None:
        hits.append(length)

    decision = decide("out", text, hits)
    _emit(decision, trace_id)
    return decision
