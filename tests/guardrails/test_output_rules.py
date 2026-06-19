"""Unit tests for the pure OUTPUT guardrail checkers (no Redis/LLM/DB)."""

from __future__ import annotations

from app.core.state import DraftResponse, ExecutionPlan, PlanStep
from app.guardrails import output_rules as orr


def _plan(tool: str, action: str, **arguments) -> ExecutionPlan:
    step = PlanStep(id="s1", tool=tool, action=action, arguments=arguments)
    return ExecutionPlan(reasoning="r", steps=[step], strategy="sequential",
                         complexity_score=1, estimated_cost_usd=0.0, requires_approval=False)


def _draft(**kw) -> DraftResponse:
    base = dict(summary="s", detail_markdown="d", actions_taken=[], actions_pending=[])
    base.update(kw)
    return DraftResponse(**base)


def test_pii_leak_redacts_and_flags_when_not_from_user():
    text = "Their SSN is 123-45-6789."
    redacted, hit = orr.check_pii_leak(text, source_was_user=False)
    assert hit is not None and hit.action == "flag_for_approval"
    assert "123-45-6789" not in redacted


def test_pii_leak_redacts_only_when_from_user():
    _, hit = orr.check_pii_leak("SSN 123-45-6789", source_was_user=True)
    assert hit is not None and hit.action == "redact"


def test_pii_leak_keeps_email_addresses_in_responses():
    # This is a mail assistant — addresses in OUTPUT are legitimate, not leaked.
    text = "I drafted an email to john@example.com."
    redacted, hit = orr.check_pii_leak(text, source_was_user=False)
    assert hit is None and "john@example.com" in redacted


def test_secret_leak_blocks():
    hit = orr.check_secret_leak("the token is AKIAIOSFODNN7EXAMPLE for the bucket")
    assert hit is not None and hit.action == "block" and hit.severity == "critical"


def test_policy_denies_tool_action():
    plan = _plan("gmail", "delete_forever")
    assert orr.check_policy_tool_args(plan, {("gmail", "delete_forever")}) is not None
    assert orr.check_policy_tool_args(plan, set()) is None


def test_destructive_delete_always_forces_approval():
    plan = _plan("calendar", "delete_event", event_id="x")
    hit = orr.check_destructive(plan, approval_actions={"calendar.delete_event"},
                                email_action="gmail.send_email", recipient_allowlist=set())
    assert hit is not None and hit.action == "flag_for_approval"


def test_email_outside_allowlist_forces_approval():
    plan = _plan("gmail", "send_email", to="ceo@external.io", subject="x", body="y")
    hit = orr.check_destructive(plan, approval_actions=set(), email_action="gmail.send_email",
                                recipient_allowlist={"example.com"})
    assert hit is not None and hit.action == "flag_for_approval"


def test_email_inside_allowlist_is_allowed():
    plan = _plan("gmail", "send_email", to="bob@example.com", subject="x", body="y")
    hit = orr.check_destructive(plan, approval_actions=set(), email_action="gmail.send_email",
                                recipient_allowlist={"example.com"})
    assert hit is None


def test_uncited_docid_flags():
    draft = _draft(citations=["plan-known", "plan-ghost"])
    hit = orr.check_hallucination_citation(draft, {"plan-known"})
    assert hit is not None and hit.action == "flag_for_approval"
    assert orr.check_hallucination_citation(draft, {"plan-known", "plan-ghost"}) is None


def test_urls_are_not_treated_as_uncited():
    draft = _draft(citations=["https://docs.example.com/x"])
    assert orr.check_hallucination_citation(draft, set()) is None


def test_profanity_redacted():
    redacted, hit = orr.check_profanity("this is crap honestly", ["crap"])
    assert hit is not None and hit.action == "redact" and "crap" not in redacted


def test_length_cap_truncates_and_warns():
    long_text = "x" * (orr.MAX_OUTPUT_TOKENS * 4 + 10)
    truncated, hit = orr.check_length(long_text)
    assert hit is not None and hit.action == "warn" and truncated.endswith("…[truncated]")
    assert orr.check_length("short")[1] is None
