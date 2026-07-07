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


def test_pii_leak_outbound_redacts_but_does_not_gate():
    # STALE (2026-06-24): outbound PII used to escalate to flag_for_approval; that escalation
    # was removed (sends are no longer HITL-gated; this scan runs post-send). PII is still
    # redacted, but it no longer gates. Old assertion kept commented per CLAUDE.md R13.
    text = "Their SSN is 123-45-6789."
    redacted, hit = orr.check_pii_leak(text, source_was_user=False, outbound=True)
    # assert hit is not None and hit.action == "flag_for_approval"
    assert hit is not None and hit.action == "redact"
    assert "123-45-6789" not in redacted


def test_pii_leak_redacts_without_gating_when_displayed_to_user():
    # Reading the user's own data back to them is not a leak: redact, don't gate.
    text = "Their SSN is 123-45-6789."
    redacted, hit = orr.check_pii_leak(text, source_was_user=False, outbound=False)
    assert hit is not None and hit.action == "redact"
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


# STALE (2026-06-23): the recipient-allowlist HITL for email sends was disabled (sends no
# longer require approval). This test asserted the retired gate; left commented per CLAUDE.md
# R13. The replacement below asserts sends are NOT gated regardless of recipient domain.
# def test_email_outside_allowlist_forces_approval():
#     plan = _plan("gmail", "send_email", to="ceo@external.io", subject="x", body="y")
#     hit = orr.check_destructive(plan, approval_actions=set(), email_action="gmail.send_email",
#                                 recipient_allowlist={"example.com"})
#     assert hit is not None and hit.action == "flag_for_approval"


def test_email_send_is_not_gated_regardless_of_recipient():
    # Sends no longer require HITL: an external recipient is NOT flagged for approval.
    external = _plan("google", "send_email", to="ceo@external.io", subject="x", body="y")
    internal = _plan("google", "send_email", to="bob@example.com", subject="x", body="y")
    for plan in (external, internal):
        hit = orr.check_destructive(plan, approval_actions=set(),
                                    email_action="google.send_email",
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


# STALE (2026-06-22): check_profanity retired (subjective, low value); this test targets a
# removed function. Left commented in place per CLAUDE.md R13.
# def test_profanity_redacted():
#     redacted, hit = orr.check_profanity("this is crap honestly", ["crap"])
#     assert hit is not None and hit.action == "redact" and "crap" not in redacted


# ── check_tool_args (google.create_event + github write paths) ───────────────
_GOOGLE_EVENT_CFG = {"max_attendees": 10, "max_duration_hours": 8}
_GITHUB_BODY_CFG = {"max_bytes": 8192}


def test_tool_args_caps_attendees():
    plan = _plan("google", "create_event", summary="s", start="2030-01-01T10:00:00",
                 end="2030-01-01T11:00:00", attendees=[f"a{i}@x.com" for i in range(11)])
    hit = orr.check_tool_args(plan, _GOOGLE_EVENT_CFG, _GITHUB_BODY_CFG)
    assert hit is not None and hit.action == "block" and "attendees" in hit.detail


def test_tool_args_caps_duration():
    # 11h event exceeds the 8h cap.
    plan = _plan("google", "create_event", summary="s", start="2030-01-01T09:00:00",
                 end="2030-01-01T20:00:00")
    hit = orr.check_tool_args(plan, _GOOGLE_EVENT_CFG, _GITHUB_BODY_CFG)
    assert hit is not None and hit.action == "block" and "duration" in hit.detail


def test_tool_args_caps_github_body():
    plan = _plan("github", "create_issue", title="t", body="x" * 9000)  # > 8192 bytes
    hit = orr.check_tool_args(plan, _GOOGLE_EVENT_CFG, _GITHUB_BODY_CFG)
    assert hit is not None and hit.action == "block" and "body too large" in hit.detail


def test_tool_args_happy_path():
    plan = _plan("google", "create_event", summary="s", start="2030-01-01T10:00:00",
                 end="2030-01-01T10:30:00", attendees=["a@x.com"])
    assert orr.check_tool_args(plan, _GOOGLE_EVENT_CFG, _GITHUB_BODY_CFG) is None


def test_is_trusted_read():
    trusted = {"github.get_pr", "finnhub.get_quote"}
    read_step = PlanStep(id="s1", tool="github", action="get_pr")
    write_step = PlanStep(id="s2", tool="reddit", action="post_comment")
    assert orr.is_trusted_read(read_step, trusted) is True
    assert orr.is_trusted_read(write_step, trusted) is False


def test_length_cap_truncates_and_warns():
    long_text = "x" * (orr.MAX_OUTPUT_TOKENS * 4 + 10)
    truncated, hit = orr.check_length(long_text)
    assert hit is not None and hit.action == "warn" and truncated.endswith("…[truncated]")
    assert orr.check_length("short")[1] is None
