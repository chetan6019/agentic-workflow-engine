"""Pure, synchronous OUTPUT guardrail checks (model/tool output → user).

Policy-dependent checks take the already-parsed policy data as arguments so these
stay pure (no file I/O); the engine loads + caches policy.yaml and the Redis-backed
mass-action throttle.
"""

from __future__ import annotations

import re

from app.core.state import DraftResponse, ExecutionPlan
from app.guardrails.contracts import Action, RuleHit
from app.security.redaction import find_secrets, redact_pii

__all__ = [
    "check_pii_leak",
    "check_secret_leak",
    "check_policy_tool_args",
    "check_destructive",
    "check_hallucination_citation",
    "check_profanity",
    "check_length",
    "MAX_OUTPUT_TOKENS",
    "recipients_of",
]

MAX_OUTPUT_TOKENS = 1_500
_EMAIL_IN = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def check_pii_leak(text: str, *, source_was_user: bool) -> tuple[str, RuleHit | None]:
    """Redact PII in a response; flag for approval when the source wasn't the user."""
    redacted, counts = redact_pii(text, include_email=False)
    if not counts:
        return text, None
    summary = ", ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
    action: Action = "redact" if source_was_user else "flag_for_approval"
    return redacted, RuleHit(rule="pii_leak", action=action, severity="medium",
                             detail=f"response contained PII ({summary}); source_was_user={source_was_user}")


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


def check_profanity(text: str, words: list[str]) -> tuple[str, RuleHit | None]:
    """Redact configured unsafe words (case-insensitive, whole word)."""
    if not words:
        return text, None
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)
    redacted, n = pattern.subn("[REDACTED]", text)
    if n:
        return redacted, RuleHit(rule="profanity", action="redact", severity="low",
                                 detail=f"redacted {n} unsafe term(s)")
    return text, None


def check_length(text: str, *, max_tokens: int = MAX_OUTPUT_TOKENS) -> tuple[str, RuleHit | None]:
    """Truncate over-long responses (≈max_tokens) and warn."""
    limit = max_tokens * 4
    if len(text) > limit:
        truncated = text[:limit] + "\n\n…[truncated]"
        return truncated, RuleHit(rule="length_cap", action="warn", severity="low",
                                  detail=f"response truncated at ~{max_tokens} tokens")
    return text, None
