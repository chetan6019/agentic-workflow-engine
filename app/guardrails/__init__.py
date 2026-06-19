"""Pure-Python input/output guardrails (allowlist/denylist + regex + Redis).

Public surface:
    evaluate_input(text, ...)  -> Decision   # user request → planner (pre-LLM)
    evaluate_output(text, ...) -> Decision   # model/tool output → user (post-LLM)
"""

from __future__ import annotations

from app.guardrails.contracts import Action, Decision, RuleHit, Severity
from app.guardrails.engine import evaluate_input, evaluate_output

__all__ = [
    "Action",
    "Decision",
    "RuleHit",
    "Severity",
    "evaluate_input",
    "evaluate_output",
]
