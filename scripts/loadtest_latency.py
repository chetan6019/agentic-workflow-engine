"""Drive a steady batch of workflow runs so p95/p99 latency becomes meaningful.

Manual one-off runs make high quantiles useless — a single slow click is the p95.
This script logs in, fires N runs at a fixed concurrency, waits for each to finish,
and prints client-side p50/p95/p99 from the collected wall-clock times. Cross-check
the server-side figure in Prometheus with the query printed at the end.

Usage (defaults in []):
    python scripts/loadtest_latency.py \
        --base-url http://localhost:8000 \
        --username demo --password demopass123 \
        --runs 200 --concurrency 5 \
        --request "list my upcoming calendar events"

Notes before you run:
  * Input rate caps apply PER USER (policy.yaml: rate_per_minute=30, rate_per_hour=300).
    200 runs will trip the hourly cap with one user — raise those caps for the test, or
    spread load across several --username accounts. The script reports any 429s it sees.
  * Every run hits real LLMs + MCP tools, so this costs tokens/money. Use a READ-only
    --request (calendar/inbox list); never a send_email prompt (side effects + 5/hr cap).
  * Concurrency choice = what you want to measure: --concurrency 1 gives the unloaded
    baseline; higher values give latency under load (contention is real, that's the point).
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


def percentile(samples: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100). Empty list returns 0.0."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = max(1, round(pct / 100.0 * len(ordered)))
    return ordered[rank - 1]


async def login(client: httpx.AsyncClient, base_url: str, username: str, password: str) -> str:
    """Return a bearer token for the load user (registers it if it doesn't exist yet)."""
    body = {"username": username, "password": password}
    r = await client.post(f"{base_url}/v1/auth/login", json=body)
    if r.status_code == 401:
        # First run on a fresh DB: create the user, then we already have a token back.
        r = await client.post(f"{base_url}/v1/auth/register", json=body)
    r.raise_for_status()
    return r.json()["access_token"]


async def one_run(client: httpx.AsyncClient, base_url: str, headers: dict,
                  request_text: str, poll_interval: float, timeout_s: float) -> dict:
    """Fire one invoke, poll until done/failed, return {ok, latency, status}."""
    start = time.monotonic()
    r = await client.post(f"{base_url}/v1/invoke", headers=headers,
                          json={"user_request": request_text})
    if r.status_code == 429:
        return {"ok": False, "latency": 0.0, "status": "rate_limited"}
    if r.status_code != 200:
        return {"ok": False, "latency": 0.0, "status": f"http_{r.status_code}"}
    trace_id = r.json()["trace_id"]

    # Poll the authoritative result until the run reaches a terminal state.
    while time.monotonic() - start < timeout_s:
        await asyncio.sleep(poll_interval)
        pr = await client.get(f"{base_url}/v1/invoke/result/{trace_id}", headers=headers)
        if pr.status_code != 200:
            continue
        status = pr.json().get("status")
        if status in ("done", "failed"):
            latency = time.monotonic() - start
            return {"ok": status == "done", "latency": latency, "status": status}
    return {"ok": False, "latency": time.monotonic() - start, "status": "client_timeout"}


async def run_load(args: argparse.Namespace) -> None:
    """Log in, fan out the runs under a concurrency cap, then report quantiles."""
    limits = httpx.Limits(max_connections=args.concurrency + 5)
    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
        token = await login(client, args.base_url, args.username, args.password)
        headers = {"Authorization": f"Bearer {token}"}
        semaphore = asyncio.Semaphore(args.concurrency)

        async def guarded() -> dict:
            async with semaphore:
                return await one_run(client, args.base_url, headers, args.request,
                                     args.poll_interval, args.run_timeout)

        wall_start = time.monotonic()
        results = await asyncio.gather(*[guarded() for _ in range(args.runs)])
        wall_elapsed = time.monotonic() - wall_start

    ok = [r for r in results if r["ok"]]
    latencies = [r["latency"] for r in ok]
    rate_limited = sum(1 for r in results if r["status"] == "rate_limited")
    failed = [r for r in results if not r["ok"] and r["status"] != "rate_limited"]

    print(f"\n=== load summary ({args.runs} runs, concurrency {args.concurrency}) ===")
    print(f"completed ok : {len(ok)}")
    print(f"rate-limited : {rate_limited}  (raise policy.yaml caps or add users if >0)")
    print(f"other failed : {len(failed)}")
    print(f"wall time    : {wall_elapsed:.1f}s  ->  {len(ok) / wall_elapsed:.2f} ok-runs/s")
    if latencies:
        print("\nclient-perceived latency (seconds):")
        print(f"  p50 : {percentile(latencies, 50):.2f}")
        print(f"  p95 : {percentile(latencies, 95):.2f}")
        print(f"  p99 : {percentile(latencies, 99):.2f}")
        print(f"  max : {max(latencies):.2f}")
        print(f"  avg : {statistics.fmean(latencies):.2f}")
    print("\nserver-side cross-check (run in Prometheus over the test window):")
    print("  histogram_quantile(0.95, sum(increase("
          "workflow_run_latency_seconds_bucket[15m])) by (le))")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latency load generator for the workflow engine.")
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--username", default="loadtest")
    p.add_argument("--password", default="loadtest-pass-123")
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--request", default="list my upcoming calendar events")
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--run-timeout", type=float, default=180.0)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_load(parse_args()))
