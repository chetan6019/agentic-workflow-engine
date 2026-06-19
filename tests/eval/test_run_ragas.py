"""Fast unit tests for the RAGAS runner's pure helpers (no ragas/LLM/Qdrant)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_MOD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "run_ragas.py"
_spec = importlib.util.spec_from_file_location("run_ragas", _MOD_PATH)
assert _spec and _spec.loader
run_ragas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_ragas)


def test_dataset_loads_and_is_well_formed():
    rows = run_ragas.load_dataset(run_ragas._DATASET)
    assert len(rows) >= 30
    for row in rows:
        assert run_ragas._REQUIRED_KEYS <= row.keys()
        assert isinstance(row["ground_truth_contexts"], list) and row["ground_truth_contexts"]


def test_load_dataset_rejects_missing_keys(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "q"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        run_ragas.load_dataset(bad)


def test_gate_requires_all_thresholds():
    assert run_ragas._gate({"faithfulness": 0.95, "context_precision": 0.85}) is True
    assert run_ragas._gate({"faithfulness": 0.95, "context_precision": 0.50}) is False
    assert run_ragas._gate({"faithfulness": 0.80, "context_precision": 0.90}) is False


def test_thresholds_match_spec():
    assert run_ragas.THRESHOLDS == {"faithfulness": 0.90, "context_precision": 0.80}
