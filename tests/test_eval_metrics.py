"""Unit tests for the offline eval metric functions (no stack required).

scripts/ is not a package, so the eval module is loaded by file path. Only the
pure recall@k / MRR / id-derivation helpers are exercised here; main() and
_search are not (they need Qdrant + the LiteLLM proxy).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "eval_retrieval", _REPO / "scripts" / "eval_retrieval.py")
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)


# ── recall@k ─────────────────────────────────────────────────────────────────


def test_recall_full_hit():
    assert ev.recall_at_k({"a"}, ["a", "b", "c"], 5) == 1.0


def test_recall_miss_outside_k():
    assert ev.recall_at_k({"a"}, ["b", "c", "d", "e", "f", "a"], 5) == 0.0


def test_recall_partial_multi_expected():
    assert ev.recall_at_k({"a", "b"}, ["a", "x", "y"], 5) == 0.5


def test_recall_empty_expected_is_zero():
    assert ev.recall_at_k(set(), ["a"], 5) == 0.0


# ── reciprocal rank ──────────────────────────────────────────────────────────


def test_rr_first_position():
    assert ev.reciprocal_rank({"a"}, ["a", "b"], 5) == 1.0


def test_rr_third_position():
    assert ev.reciprocal_rank({"c"}, ["a", "b", "c"], 5) == pytest.approx(1 / 3)


def test_rr_not_found_in_k_is_zero():
    assert ev.reciprocal_rank({"z"}, ["a", "b", "c", "d", "e", "z"], 5) == 0.0


def test_rr_uses_first_relevant_only():
    assert ev.reciprocal_rank({"b", "c"}, ["a", "b", "c"], 5) == pytest.approx(1 / 2)


# ── seed id derivation must match the seeder + golden set ────────────────────


def test_seed_id_matches_seed_script_formula():
    import uuid
    req = "Show my unread emails"
    assert ev._seed_plan_id(req) == str(uuid.uuid5(uuid.NAMESPACE_DNS, f"demo-plan::{req}"))


def test_golden_set_is_well_formed():
    golden = ev._load_golden(_REPO / "tests" / "golden_retrieval.jsonl")
    assert len(golden) >= 20
    for entry in golden:
        assert entry["query"].strip()
        assert entry["expected_seed_requests"]
        assert all(isinstance(r, str) and r.strip() for r in entry["expected_seed_requests"])
