"""Pure, synchronous OUTPUT guardrail checks (model/tool output → user).

Policy-dependent checks take the already-parsed policy data as arguments so these
stay pure (no file I/O); the engine loads + caches policy.yaml and the Redis-backed
mass-action throttle.
"""

from __future__ import annotations

import datetime
import re

from app.core.state import DraftResponse, ExecutionPlan, PlanStep
from app.guardrails.contracts import Action, RuleHit
from app.security.redaction import find_secrets, redact_pii

__all__ = [
    "check_pii_leak",
    "check_secret_leak",
    "check_policy_tool_args",
    "check_tool_args",
    "check_destructive",
    "check_hallucination_citation",
    # STALE (2026-06-22): check_profanity retired (subjective, low value).
    # "check_profanity",
    "check_length",
    "is_trusted_read",
    "MAX_OUTPUT_TOKENS",
    "recipients_of",
]

MAX_OUTPUT_TOKENS = 1_500
_EMAIL_IN = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def check_pii_leak(text: str, *, source_was_user: bool,
                   outbound: bool = False) -> tuple[str, RuleHit | None]:
    """Redact PII in a response; flag for approval only when it would actually leak.

    PII is always redacted from the shown text. It escalates to ``flag_for_approval``
    only when the response is OUTBOUND (an email send leaving the user's trust
    boundary) and the PII didn't originate from the user. Displaying the user's own
    data back to them — reading their inbox, calendar, etc. — is not a leak, so it
    redacts without gating. Outbound exfiltration is also covered by check_destructive.
    """
    redacted, counts = redact_pii(text, include_email=False)
    if not counts:
        return text, None
    summary = ", ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
    escalate = outbound and not source_was_user
    action: Action = "flag_for_approval" if escalate else "redact"
    return redacted, RuleHit(rule="pii_leak", action=action, severity="medium",
                             detail=f"response contained PII ({summary}); "
                                    f"source_was_user={source_was_user}, outbound={outbound}")


def check_secret_leak(text: str) -> RuleHit | None:
    """Block (and let the engine audit) any secret leaking into the response."""
    kinds = find_secrets(text)
    if kinds:
        return RuleHit(rule="secret_leak", action="block", severity="critical",
                       detail=f"response leaked secret-shaped tokens: {', '.join(sorted(kinds))}")
    return None


def check_policy_tool_args(plan: ExecutionPlan | None,
                           denied: set[tuple[str, str]]) -> RuleHit | None:
    """Block when any final plan step matches a denied (tool, action) pair."""
    if plan is None:
        return None
    for step in plan.steps:
        if (step.tool, step.action) in denied:
            return RuleHit(rule="policy_tool_args", action="block", severity="high",
                           detail=f"denied tool action: {step.tool}.{step.action}")
    return None


def _cfg_number(value: object, default: float) -> float:
    """Read a numeric policy value, falling back to ``default`` for missing/non-numeric ones.

    Policy dicts are typed ``dict[str, object]`` (YAML can hold mixed types), so we narrow
    to a real number here instead of trusting the value blindly. Beginner-friendly and pure.
    """
    if isinstance(value, bool):  # bool is an int subclass; a flag is never a numeric cap
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _event_duration_hours(arguments: dict[str, object]) -> float | None:
    """Return a calendar event's duration in hours from ISO start/end args, else None.

    Pure: no I/O. Returns None when start/end are missing or not parseable ISO strings,
    so a malformed arg is simply not duration-checked here (other rules still apply).
    """
    start = arguments.get("start")
    end = arguments.get("end")
    if not (isinstance(start, str) and isinstance(end, str)):
        return None
    try:
        start_dt = datetime.datetime.fromisoformat(start)
        end_dt = datetime.datetime.fromisoformat(end)
    except ValueError:
        return None
    return (end_dt - start_dt).total_seconds() / 3600.0


def check_tool_args(plan: ExecutionPlan | None, google_event_cfg: dict[str, object],
                    github_body_cfg: dict[str, object]) -> RuleHit | None:
    """Block tool steps whose arguments break a tool-specific sanity cap.

    Beginner-friendly: one explicit ``if`` per guarded (tool, action) — no dispatch tables.
    Covers google.create_event (attendee + duration caps) and the two github write paths
    (body-size cap). Past-start checking for google.create_event is the CALLER's job: the
    ``deny_past_start`` policy flag is honoured by the calendar tool, not this pure function.
    """
    if plan is None:
        return None
    max_attendees = int(_cfg_number(google_event_cfg.get("max_attendees"), 10))
    max_duration_hours = _cfg_number(google_event_cfg.get("max_duration_hours"), 8)
    max_body_bytes = int(_cfg_number(github_body_cfg.get("max_bytes"), 8192))
    for step in plan.steps:
        if step.tool == "google" and step.action == "create_event":
            attendees = step.arguments.get("attendees")
            if isinstance(attendees, list) and len(attendees) > max_attendees:
                return RuleHit(rule="tool_args", action="block", severity="high",
                               detail=f"google.create_event: too many attendees "
                                      f"({len(attendees)} > {max_attendees})")
            hours = _event_duration_hours(step.arguments)
            if hours is not None and hours > max_duration_hours:
                return RuleHit(rule="tool_args", action="block", severity="high",
                               detail=f"google.create_event: duration too long "
                                      f"({hours:.1f}h > {max_duration_hours}h)")
        if step.tool == "github" and step.action in ("create_issue", "comment_on_pr"):
            body = step.arguments.get("body")
            if isinstance(body, str) and len(body.encode("utf-8")) > max_body_bytes:
                return RuleHit(rule="tool_args", action="block", severity="medium",
                               detail=f"{step.tool}.{step.action}: body too large "
                                      f"({len(body.encode('utf-8'))} bytes > {max_body_bytes})")
    return None


def recipients_of(arguments: dict[str, object]) -> list[str]:
    """Pull recipient email domains out of a tool step's arguments."""
    raw: list[str] = []
    for key in ("to", "recipient", "cc", "bcc"):
        value = arguments.get(key)
        if isinstance(value, str):
            raw.append(value)
        elif isinstance(value, list):
            raw.extend(v for v in value if isinstance(v, str))
    return [m.group(1).lower() for addr in raw for m in [_EMAIL_IN.search(addr)] if m]


def check_destructive(plan: ExecutionPlan | None, *, approval_actions: set[str],
                      email_action: str, recipient_allowlist: set[str]) -> RuleHit | None:
    """Force approval for deletes, and for emails to non-allowlisted domains."""
    if plan is None:
        return None
    for step in plan.steps:
        key = f"{step.tool}.{step.action}"
        if key in approval_actions:
            return RuleHit(rule="destructive_action", action="flag_for_approval", severity="high",
                           detail=f"destructive action requires approval: {key}")
        if key == email_action:
            external = [d for d in recipients_of(step.arguments) if d not in recipient_allowlist]
            if external:
                return RuleHit(rule="destructive_action", action="flag_for_approval", severity="high",
                               detail=f"email to non-allowlisted domain(s): {', '.join(sorted(set(external)))}")
    return None


def check_hallucination_citation(draft: DraftResponse | None,
                                 retrieved_ids: set[str]) -> RuleHit | None:
    """Flag when the draft cites a doc/plan id that wasn't in the retrieved context."""
    if draft is None:
        return None
    unknown = [c for c in draft.citations
               if not c.startswith(("http://", "https://")) and c not in retrieved_ids]
    if unknown:
        return RuleHit(rule="hallucination_citation", action="flag_for_approval", severity="medium",
                       detail=f"uncited source id(s): {', '.join(sorted(set(unknown)))}")
    return None


# STALE (2026-06-22): check_profanity retired — a hand-maintained wordlist is a subjective,
# low-value signal. Left in place per CLAUDE.md R13; policy.yaml's profanity list is now empty
# and the engine no longer calls this.
# def check_profanity(text: str, words: list[str]) -> tuple[str, RuleHit | None]:
#     """Redact configured unsafe words (case-insensitive, whole word)."""
#     if not words:
#         return text, None
#     pattern = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)
#     redacted, n = pattern.subn("[REDACTED]", text)
#     if n:
#         return redacted, RuleHit(rule="profanity", action="redact", severity="low",
#                                  detail=f"redacted {n} unsafe term(s)")
#     return text, None


def is_trusted_read(step: PlanStep, trusted_actions: set[str]) -> bool:
    """True iff this step's ``tool.action`` is in the trusted-read carve-out set."""
    return f"{step.tool}.{step.action}" in trusted_actions


def check_length(text: str, *, max_tokens: int = MAX_OUTPUT_TOKENS) -> tuple[str, RuleHit | None]:
    """Truncate over-long responses (≈max_tokens) and warn."""
    limit = max_tokens * 4
    if len(text) > limit:
        truncated = text[:limit] + "\n\n…[truncated]"
        return truncated, RuleHit(rule="length_cap", action="warn", severity="low",
                                  detail=f"response truncated at ~{max_tokens} tokens")
    return text, None
