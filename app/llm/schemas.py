"""PII redaction helpers — for log/trace copies only, never live LLM input."""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"\b(?:\+?\d{1,2}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b")

_PATTERNS = [_EMAIL, _PHONE, _SSN, _CARD]


def redact_pii(text: str) -> str:
    """Replace emails, phone numbers, SSNs, and credit-card numbers with [REDACTED]."""
    for pat in _PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text
