"""Prometheus metrics for the workflow engine.

Kept tiny and import-safe: defining a Counter just registers it on the default
registry. The /metrics endpoint (app/api/health.py) exposes the registry.
"""

from __future__ import annotations

from prometheus_client import Counter

# Incremented whenever a guardrail blocks, flags-for-approval, or rate-limits a
# request. ``direction`` is "in" (user → planner) or "out" (model → user).
guardrail_block_total: Counter = Counter(
    "guardrail_block_total",
    "Guardrail enforcement actions by rule and direction.",
    ["rule", "direction"],
)
