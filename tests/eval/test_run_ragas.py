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


class _FakeLangfuse:
    """Captures the eval trace + scores pushed by _push_scores_to_langfuse."""

    instances: list["_FakeLangfuse"] = []

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.scores: list[dict] = []
        self.flushed = False
        _FakeLangfuse.instances.append(self)

    def create_trace_id(self) -> str:
        return "trace-1"

    def create_event(self, **kwargs) -> None:
        self.events.append(kwargs)

    def create_score(self, **kwargs) -> None:
        self.scores.append(kwargs)

    def flush(self) -> None:
        self.flushed = True


def test_push_scores_skipped_without_keys(monkeypatch):
    """No LANGFUSE_* env keys → no client is ever constructed (no import side effects)."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    _FakeLangfuse.instances.clear()
    run_ragas._push_scores_to_langfuse({"faithfulness": 0.95}, "abc123", 3)
    assert _FakeLangfuse.instances == []


def test_push_scores_creates_trace_and_per_metric_scores(monkeypatch):
    import sys
    import types

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    fake_mod = types.ModuleType("langfuse")
    fake_mod.Langfuse = _FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", fake_mod)
    _FakeLangfuse.instances.clear()

    scores = {"faithfulness": 0.95, "context_precision": 0.85}
    run_ragas._push_scores_to_langfuse(scores, "abc123", 3)

    client = _FakeLangfuse.instances[0]
    assert client.events[0]["name"] == "ragas-eval"
    assert client.events[0]["trace_context"] == {"trace_id": "trace-1"}
    assert client.events[0]["output"] == scores
    pushed = {s["name"]: s["value"] for s in client.scores}
    assert pushed == {"ragas_faithfulness": 0.95, "ragas_context_precision": 0.85}
    assert all(s["trace_id"] == "trace-1" for s in client.scores)
    assert client.flushed is True
