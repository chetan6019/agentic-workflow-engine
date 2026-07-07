"""Pure, synchronous INPUT guardrail checks (user request → planner).

Each checker takes text and returns a RuleHit or None. Redaction-style checks
return the rewritten text too. Redis-backed rate caps live in the engine (async).
"""

from __future__ import annotations

import base64
import re

from app.guardrails.contracts import RuleHit
from app.security.redaction import find_secrets, redact_pii

__all__ = [
    "check_prompt_injection",
    "check_jailbreak",
    "check_secrets",
    "check_pii",
    "check_oversized",
    # STALE (2026-06-22): check_language retired (vague signal, low value).
    # "check_language",
    "MAX_CHARS",
    "MAX_TOKENS",
]

MAX_CHARS = 12_000
MAX_TOKENS = 4_000  # ~chars/4 heuristic; avoids a tokenizer dependency

_INJECTION = re.compile(
    r"(?:ignore|disregard|forget)\s+(?:all\s+|the\s+|your\s+)?(?:previous|prior|above|earlier"
    r"|past)?\s*(?:instructions|rules|prompt|guidelines|context)"
    r"|you\s+are\s+now\b"
    r"|(?:reveal|print|show|repeat|output|leak)\s+(?:me\s+)?(?:your\s+|the\s+)?(?:system\s+)?prompt"
    r"|new\s+instructions\s*:",
    re.IGNORECASE,
)
_JAILBREAK = re.compile(
    r"\bDAN\b|do\s+anything\s+now|developer\s+mode|jailbreak"
    r"|as\s+an\s+AI\s+(?:model\s+)?without\s+(?:any\s+)?restrictions"
    r"|without\s+any\s+(?:restrictions|filters|guardrails)",
    re.IGNORECASE,
)
# Long base64-ish blob that may smuggle an instruction past the plain-text scan.
_B64 = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


def check_prompt_injection(text: str) -> RuleHit | None:
    """Block classic prompt-injection / system-prompt-exfiltration phrasing."""
    if _INJECTION.search(text):
        return RuleHit(rule="prompt_injection", action="block", severity="high",
                       detail="request attempts to override or exfiltrate instructions")
    return None


def _b64_smuggles(text: str) -> bool:
    """True if any base64 blob decodes to injection/jailbreak text."""
    for match in _B64.finditer(text):
        try:
            decoded = base64.b64decode(match.group(0), validate=True).decode("utf-8", "ignore")
        except (ValueError, UnicodeDecodeError):
            continue
        if _INJECTION.search(decoded) or _JAILBREAK.search(decoded):
            return True
    return False


def check_jailbreak(text: str) -> RuleHit | None:
    """Block known jailbreak phrases and base64-smuggled instructions."""
    if _JAILBREAK.search(text) or _b64_smuggles(text):
        return RuleHit(rule="jailbreak", action="block", severity="high",
                       detail="request matches a known jailbreak / restriction-bypass pattern")
    return None


def check_secrets(text: str) -> RuleHit | None:
    """Block requests carrying secret/API-key shapes (raw value never logged)."""
    kinds = find_secrets(text)
    if kinds:
        return RuleHit(rule="secret_shapes", action="block", severity="critical",
                       detail=f"input contains secret-shaped tokens: {', '.join(sorted(kinds))}")
    return None


def check_pii(text: str) -> tuple[str, RuleHit | None]:
    """Redact PII before the LLM sees it; report categories + counts, never values.

    Email addresses are KEPT (include_email=False): this is a mail assistant, so the
    recipient address the user types is essential functional data the planner needs —
    redacting it to [REDACTED] breaks every send request. Other PII (SSN, phone,
    cards) is still scrubbed. Mirrors the output rule's reasoning.
    """
    redacted, counts = redact_pii(text, include_email=False)
    if counts:
        summary = ", ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
        return redacted, RuleHit(rule="pii_redact", action="redact", severity="medium",
                                 detail=f"redacted PII before planning: {summary}")
    return redacted, None


def check_oversized(text: str) -> RuleHit | None:
    """Block requests over the character or estimated-token limit."""
    if len(text) > MAX_CHARS or (len(text) // 4) > MAX_TOKENS:
        return RuleHit(rule="oversized_input", action="block", severity="medium",
                       detail=f"input too large ({len(text)} chars; limit {MAX_CHARS})")
    return None


# STALE (2026-06-22): check_language retired — a low-ASCII-ratio heuristic is a vague,
# low-value signal that misfires on legitimate non-English input. Left in place per
# CLAUDE.md R13 (comment, don't delete); no longer imported or called by the engine.
# def check_language(text: str) -> RuleHit | None:
#     """Flag (don't block) input that doesn't look like supported English text."""
#     letters = [c for c in text if c.isalpha()]
#     if len(letters) < 10:
#         return None
#     ascii_ratio = sum(c.isascii() for c in letters) / len(letters)
#     if ascii_ratio < 0.5:
#         return RuleHit(rule="unsupported_language", action="warn", severity="low",
#                        detail="input appears to be in an unsupported language")
#     return None
