"""Offline retrieval-quality eval: recall@5 and MRR for hybrid vs dense search.

Requires the running stack — Qdrant seeded via scripts/seed_demo_data.py, the
LiteLLM proxy for dense embeddings, and fastembed for sparse. Run with:

    uv run python scripts/eval_retrieval.py

It searches each query in tests/golden_retrieval.jsonl under
HYBRID_SEARCH_ENABLED on and off (the collection carries both vectors, so no
re-index is needed) and prints recall@5 / MRR per mode, making the hybrid
toggle's value measurable. Not part of CI — the pure metric functions below are
unit-tested in tests/test_eval_metrics.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import get_settings  # noqa: E402
from app.rag.retriever import _TOP_N, _search  # noqa: E402

_GOLDEN_PATH = Path(__file__).resolve().parent.parent / "tests" / "golden_retrieval.jsonl"


def recall_at_k(expected: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of expected plan ids present in the top-k retrieved ids."""
    if not expected:
        return 0.0
    return len(expected & set(retrieved[:k])) / len(expected)


def reciprocal_rank(expected: set[str], retrieved: list[str], k: int) -> float:
    """1/rank of the first relevant id in the top-k, else 0.0 (for MRR)."""
    for rank, pid in enumerate(retrieved[:k], start=1):
        if pid in expected:
            return 1.0 / rank
    return 0.0


def _seed_plan_id(seed_request: str) -> str:
    """Derive a seeded plan's id from its request text.

    Must mirror scripts/seed_demo_data.py exactly; keeping it a pure function of
    the request text is why the golden set labels by seed request, not raw uuid.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"demo-plan::{seed_request}"))


def _load_golden(path: Path) -> list[dict]:
    """Read the golden jsonl into a list of {query, expected_seed_requests}."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def _eval_mode(hybrid: bool, golden: list[dict], k: int) -> tuple[float, float]:
    """Run every golden query under one hybrid setting; return (recall@k, MRR)."""
    os.environ["HYBRID_SEARCH_ENABLED"] = "true" if hybrid else "false"
    get_settings.cache_clear()  # re-read the flipped flag
    recalls: list[float] = []
    rrs: list[float] = []
    for entry in golden:
        expected = {_seed_plan_id(r) for r in entry["expected_seed_requests"]}
        plans = await _search(entry["query"], {"success": True}, k=k)
        retrieved = [p.plan_id for p in plans]
        recalls.append(recall_at_k(expected, retrieved, k))
        rrs.append(reciprocal_rank(expected, retrieved, k))
    return mean(recalls), mean(rrs)


async def main() -> None:
    """Evaluate hybrid vs dense retrieval over the golden set and print a table."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _GOLDEN_PATH
    golden = _load_golden(path)
    print(f"Golden set: {len(golden)} queries from {path.name}\n")

    try:
        hybrid_recall, hybrid_mrr = await _eval_mode(True, golden, _TOP_N)
    except Exception as exc:  # noqa: BLE001 — offline tool: report and bail clearly
        print(f"ERROR: retrieval failed — is the stack up (Qdrant seeded, LiteLLM "
              f"proxy reachable)?\n  {type(exc).__name__}: {exc}")
        sys.exit(1)
    dense_recall, dense_mrr = await _eval_mode(False, golden, _TOP_N)

    print(f"{'mode':<10}{'recall@' + str(_TOP_N):<12}{'MRR':<8}")
    print(f"{'-' * 28}")
    print(f"{'hybrid':<10}{hybrid_recall:<12.3f}{hybrid_mrr:<8.3f}")
    print(f"{'dense':<10}{dense_recall:<12.3f}{dense_mrr:<8.3f}")
    print(f"\nΔ recall@{_TOP_N}: {hybrid_recall - dense_recall:+.3f}   "
          f"Δ MRR: {hybrid_mrr - dense_mrr:+.3f}  (hybrid − dense)")


if __name__ == "__main__":
    asyncio.run(main())
