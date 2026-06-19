"""Frozen contract types shared by the input and output guardrail pipelines."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Action", "Severity", "RuleHit", "Decision", "decide"]

# Strongest-first ordering is encoded in `_RANK` below; the engine folds many hits
# into one decision by taking the strongest action seen.
Action = Literal["block", "flag_for_approval", "rate_limit", "redact", "warn", "allow"]
Severity = Literal["low", "medium", "high", "critical"]

_RANK: dict[Action, int] = {
    "block": 5,
    "flag_for_approval": 4,
    "rate_limit": 3,
    "redact": 2,
    "warn": 1,
    "allow": 0,
}


class RuleHit(BaseModel):
    """One guardrail rule firing on a request or response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str = Field(description="Stable rule name, e.g. 'prompt_injection'.")
    action: Action = Field(description="What the engine did about it.")
    severity: Severity = Field(description="Relative risk of the hit.")
    detail: str = Field(description="Safe, value-free description of the hit.")


class Decision(BaseModel):
    """Folded result of running a rule list over one piece of text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: Literal["in", "out"] = Field(description="'in' (input) or 'out' (output).")
    allowed: bool = Field(description="False when any rule blocked the request.")
    text: str = Field(description="The (possibly redacted/truncated) text to use downstream.")
    requires_approval: bool = Field(description="True when a rule routes to HITL approval.")
    hits: tuple[RuleHit, ...] = Field(default=(), description="Every rule that fired.")

    @property
    def blocked_rules(self) -> tuple[str, ...]:
        """Names of the rules that blocked, for error messages."""
        return tuple(h.rule for h in self.hits if h.action == "block")


def decide(direction: Literal["in", "out"], text: str, hits: list[RuleHit]) -> Decision:
    """Fold a list of rule hits into a single Decision (strongest action wins)."""
    strongest = max((h.action for h in hits), key=lambda a: _RANK[a], default="allow")
    return Decision(
        direction=direction,
        allowed=strongest != "block",
        text=text,
        requires_approval=any(h.action == "flag_for_approval" for h in hits),
        hits=tuple(hits),
    )
