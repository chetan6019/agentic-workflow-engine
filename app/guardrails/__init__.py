"""Pure-Python input/output guardrails (allowlist/denylist + regex + Redis).

Public surface:
    evaluate_input(text, ...)  -> Decision   # user request → planner (pre-LLM)
    evaluate_output(text, ...) -> Decision   # model/tool output → user (post-LLM)
    approval_required_actions() -> set[str]  # "tool.action" writes gated pre-execution
    trusted_read_actions() -> set[str]       # reads exempt from the confidence gate
    confidence_gate_threshold() -> str       # "medium" | "low" | "off"
"""

from __future__ import annotations

from app.guardrails.contracts import Action, Decision, RuleHit, Severity
from app.guardrails.engine import (
    approval_required_actions,
    confidence_gate_threshold,
    evaluate_input,
    evaluate_output,
    trusted_read_actions,
)

__all__ = [
    "Action",
    "Decision",
    "RuleHit",
    "Severity",
    "approval_required_actions",
    "confidence_gate_threshold",
    "evaluate_input",
    "evaluate_output",
    "trusted_read_actions",
]
