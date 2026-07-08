"""Prometheus metrics for the workflow engine.

Kept tiny and import-safe: defining a Counter just registers it on the default
registry. The /metrics endpoint (app/api/health.py) exposes the registry.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# Incremented whenever a guardrail blocks, flags-for-approval, or rate-limits a
# request. ``direction`` is "in" (user → planner) or "out" (model → user).
guardrail_block_total: Counter = Counter(
    "guardrail_block_total",
    "Guardrail enforcement actions by rule and direction.",
    ["rule", "direction"],
)

# Shared latency buckets in SECONDS. Tuned for LLM-workflow timings: sub-second tool
# calls up to multi-minute runs near the run_timeout ceiling. Prometheus reads the
# bucket boundaries to compute quantiles (histogram_quantile) and rates in Grafana.
# The 10-30-60-120 region is deliberately dense: a full agentic run (several sequential
# LLM round-trips) lands there, so without 15/20/45/90 boundaries histogram_quantile
# interpolates p95/p99 across 20-60s-wide gaps and pins the estimate near the bucket top.
# Changing these boundaries RESETS the existing histogram series (old cumulative counts
# are discarded under the new layout) — expect a one-time gap after deploy.
_LATENCY_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0
)

# End-to-end run latency. ``outcome`` is "ok" | "error" | "timeout". The histogram's
# _count series also gives run throughput (rate) and error ratio per outcome.
workflow_run_latency_seconds: Histogram = Histogram(
    "workflow_run_latency_seconds",
    "End-to-end workflow run latency in seconds, by outcome.",
    ["outcome"],
    buckets=_LATENCY_BUCKETS,
)

# node_latency_seconds / mcp_tool_latency_seconds / request_duration_seconds removed —
# these latencies now come from OTel spans (node spans, mcp.tool_call spans, and
# FastAPI auto-instrumentation), derived to Prometheus by the collector.

# ---- Per-role LLM token + cost counters ----
# App-side fallback for LiteLLM's proxy /metrics (which on some builds is gated
# behind an enterprise key). ``role`` is the LiteLLM router alias (planner-default,
# composer, retriever-grader, guard-judge, ...). ``model`` is the actual model
# resolved by LiteLLM (e.g. groq/openai/gpt-oss-120b). ``kind`` separates prompt
# from completion tokens so you can graph prompt-vs-completion mix per role.
llm_call_tokens_total: Counter = Counter(
    "llm_call_tokens_total",
    "Tokens consumed by LLM calls, summed across calls.",
    ["role", "model", "kind"],
)
llm_call_cost_usd_total: Counter = Counter(
    "llm_call_cost_usd_total",
    "Dollar cost of LLM calls, summed across calls (USD).",
    ["role", "model"],
)

# ---- Planner outcome counter (for "plan validity" resume metric) ----
# outcome:
#   valid          -> first planner attempt passed find_invalid_plan_steps.
#   self_corrected -> first attempt invalid; the in-node correction re-ask fixed it.
#   failed         -> generate_plan_with_retry or validate_and_fix_plan raised
#                     RuntimeError (both attempts exhausted).
# First-try validity = valid / (valid + self_corrected + failed)
# End-to-end validity = (valid + self_corrected) / total
planner_outcome_total: Counter = Counter(
    "planner_outcome_total",
    "Counts of planner outcomes per call.",
    ["outcome"],
)

# ---- Guard verdict counter (for "auto-approval rate" resume metric) ----
# Values mirror state.verdict set by guard.decide_verdict so the metric reads
# the same vocabulary as the rest of the codebase, not a separate dialect.
# outcome:
#   finalize -> confidence >= HIGH_CONFIDENCE, no HITL needed -> auto-execute.
#   hitl     -> medium confidence AND HITL_ENABLED -> sent to approval queue.
#   replan   -> low confidence, retry budget remaining -> back to planner.
#   block    -> low confidence, retry budget exhausted -> hard refuse.
# Auto-approval rate = finalize / (finalize + hitl + replan + block)
guard_outcome_total: Counter = Counter(
    "guard_outcome_total",
    "Counts of guard verdicts per draft.",
    ["outcome"],
)
