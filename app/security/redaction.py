"""Shared PII / secret detection and redaction — pure regex + Luhn, no ML.

Used by the guardrails input/output pipelines (and re-exported into the guard
node so the deterministic draft-PII check shares one set of patterns). Secret
*values* are never returned or logged — only the kind that matched.
"""

from __future__ import annotations

import re

__all__ = [
    "PLACEHOLDER",
    "luhn_valid",
    "redact_pii",
    "contains_pii",
    "find_secrets",
]

PLACEHOLDER = "[REDACTED]"

# PII patterns. Emails/phones are redacted in user input but, per the guard node's
# long-standing note, are inherent to a mail/calendar assistant's *output*; the
# output pipeline only flags them when the source was not the user themselves.
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"\b(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]?\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Candidate 13–16 digit run (optionally space/dash grouped); confirmed via Luhn so
# random long numbers (ids, timestamps) don't get falsely redacted as cards.
_CARD = re.compile(r"\b(?:\d[ -]?){13,16}\b")

# Secret/API-key shapes. Keys are names only — the matched text is never surfaced.
_SECRETS: dict[str, re.Pattern[str]] = {
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_pat": re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
}


def luhn_valid(number: str) -> bool:
    """Return True if the digit run passes the Luhn checksum (credit-card shape)."""
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _redact_cards(text: str, counts: dict[str, int]) -> str:
    """Replace only Luhn-valid card-shaped runs; count the redactions."""
    def _sub(match: re.Match[str]) -> str:
        if luhn_valid(match.group(0)):
            counts["credit_card"] = counts.get("credit_card", 0) + 1
            return PLACEHOLDER
        return match.group(0)

    return _CARD.sub(_sub, text)


def redact_pii(text: str, *, include_email: bool = True) -> tuple[str, dict[str, int]]:
    """Redact SSNs, Luhn-valid cards, phones (and emails); return (text, counts).

    ``counts`` maps category → number of redactions, so callers can log how much
    was scrubbed without ever holding the raw values. ``include_email`` is False on
    the output path: this is a mail/calendar assistant, so addresses in a *response*
    are legitimate — only genuinely sensitive PII (SSN/card/phone) is scrubbed there.
    """
    counts: dict[str, int] = {}
    text = _redact_cards(text, counts)
    patterns = [("ssn", _SSN), ("phone", _PHONE)]
    if include_email:
        patterns.append(("email", _EMAIL))
    for name, pattern in patterns:
        text, n = pattern.subn(PLACEHOLDER, text)
        if n:
            counts[name] = counts.get(name, 0) + n
    return text, counts


def contains_pii(text: str) -> bool:
    """True if any sensitive PII (SSN or Luhn-valid card) is present.

    Deliberately narrow (no email/phone) — this backs the guard node's draft
    block, where addresses are legitimate assistant output.
    """
    if _SSN.search(text):
        return True
    return any(luhn_valid(m.group(0)) for m in _CARD.finditer(text))


def find_secrets(text: str) -> list[str]:
    """Return the kinds of secret present (e.g. ['openai_key']); never the values."""
    return [name for name, pattern in _SECRETS.items() if pattern.search(text)]
