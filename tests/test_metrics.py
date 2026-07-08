"""Tests for the Prometheus metrics: histograms register and observe without error.

No scraping/HTTP — we exercise the metric objects directly and confirm they render into the
default registry that /metrics (app/api/health.py) exposes.
node_latency_seconds / mcp_tool_latency_seconds / request_duration_seconds were removed —
those latencies now come from OTel spans (see tests/test_tracing.py).
"""

from __future__ import annotations

from prometheus_client import generate_latest

from app.core import metrics


def test_run_latency_histogram_observes_and_renders():
    metrics.workflow_run_latency_seconds.labels(outcome="ok").observe(0.2)
    metrics.workflow_run_latency_seconds.labels(outcome="timeout").observe(120.0)

    rendered = generate_latest().decode()
    # The metric exposes the _bucket/_count series Grafana's PromQL relies on.
    assert "workflow_run_latency_seconds_bucket" in rendered
    assert "workflow_run_latency_seconds_count" in rendered
    # Label values are present so the dashboard's by-label aggregations work.
    assert 'outcome="timeout"' in rendered


def test_removed_histograms_are_gone():
    # Guard against the hand-rolled latency histograms silently coming back —
    # they were replaced by OTel span-derived metrics.
    assert not hasattr(metrics, "node_latency_seconds")
    assert not hasattr(metrics, "mcp_tool_latency_seconds")
    assert not hasattr(metrics, "request_duration_seconds")


def test_guardrail_counter_still_present():
    metrics.guardrail_block_total.labels(rule="prompt_injection", direction="in").inc()
    assert "guardrail_block_total" in generate_latest().decode()
