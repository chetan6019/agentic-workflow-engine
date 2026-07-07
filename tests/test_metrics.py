"""Tests for the Prometheus metrics: histograms register and observe without error.

No scraping/HTTP — we exercise the metric objects directly and confirm they render into the
default registry that /metrics (app/api/health.py) exposes.
"""

from __future__ import annotations

from prometheus_client import generate_latest

from app.core import metrics


def test_latency_histograms_observe_and_render():
    metrics.workflow_run_latency_seconds.labels(outcome="ok").observe(0.2)
    metrics.workflow_run_latency_seconds.labels(outcome="timeout").observe(120.0)
    metrics.node_latency_seconds.labels(node="planner").observe(0.5)
    metrics.mcp_tool_latency_seconds.labels(server="finnhub", ok="true").observe(0.05)
    metrics.mcp_tool_latency_seconds.labels(server="github", ok="false").observe(1.0)

    rendered = generate_latest().decode()
    # Each metric exposes the _bucket/_count series Grafana's PromQL relies on.
    assert "workflow_run_latency_seconds_bucket" in rendered
    assert "workflow_run_latency_seconds_count" in rendered
    assert "node_latency_seconds_bucket" in rendered
    assert "mcp_tool_latency_seconds_bucket" in rendered
    # Label values are present so the dashboard's by-label aggregations work.
    assert 'outcome="timeout"' in rendered
    assert 'server="github"' in rendered


def test_guardrail_counter_still_present():
    metrics.guardrail_block_total.labels(rule="prompt_injection", direction="in").inc()
    assert "guardrail_block_total" in generate_latest().decode()
