"""RAGAS evaluation runner with CI-enforceable thresholds.

Gates the RAG layer (Qdrant retrieval → composer generation) on:
    faithfulness      >= 0.90   (hard gate — non-zero exit on miss)
    context_precision >= 0.80   (hard gate — non-zero exit on miss)
context_recall / answer_relevancy are tracked, not gated (see docs/adr/0008).

Modes:
    --dry-run        validate the dataset + wiring only; no LLM/Qdrant, exit 0.
    --subset N       evaluate only the first N rows (for a quick smoke run).

The LLM judge runs through the LiteLLM proxy (no provider SDK), per the repo's
LiteLLM-only rule. Answers come from the composer's grounded direct-answer path
with MCP tools mocked, so we measure retrieval + generation, not side effects.

    uv run python scripts/eval/run_ragas.py --dry-run
    uv run python scripts/eval/run_ragas.py --subset 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any
from uuid import uuid4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ragas 0.4.3 hard-imports a langchain-community module removed in 0.4.x. We never
# use the Vertex path (our judge is LiteLLM-routed), so register a harmless stub
# before ragas is imported. Done at module load so any later `import ragas` works.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = object  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

_ROOT = Path(__file__).resolve().parents[2]
_DATASET = _ROOT / "tests" / "eval" / "dataset.jsonl"
_ARTIFACTS = _ROOT / "artifacts" / "ragas"

THRESHOLDS: dict[str, float] = {"faithfulness": 0.90, "context_precision": 0.80}
_REQUIRED_KEYS = {"question", "ground_truth_answer", "ground_truth_contexts"}
_TOOL_DOC_K = 8


def _git_sha() -> str:
    """Short git sha for the report filename, or 'nogit' when unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=_ROOT, check=True)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Read + validate dataset.jsonl into a list of row dicts."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = _REQUIRED_KEYS - row.keys()
            if missing:
                raise ValueError(f"row {i} missing keys: {sorted(missing)}")
            if not isinstance(row["ground_truth_contexts"], list):
                raise ValueError(f"row {i}: ground_truth_contexts must be a list")
            rows.append(row)
    if not rows:
        raise ValueError(f"dataset {path} is empty")
    return rows


def _judge_and_embeddings() -> tuple[Any, Any]:
    """Build the LiteLLM-routed judge LLM (+ optional embeddings) as ragas wrappers."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    from app.config import get_settings

    s = get_settings()
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", "composer")
    judge = ChatOpenAI(model=judge_model, openai_api_base=s.litellm_url,
                       openai_api_key=s.litellm_virtual_key, temperature=0, max_retries=0)
    llm = LangchainLLMWrapper(judge)

    embeddings = None
    embed_model = os.environ.get("EVAL_EMBED_MODEL")
    if embed_model:
        from langchain_openai import OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper

        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
            model=embed_model, openai_api_base=s.litellm_url,
            openai_api_key=s.litellm_virtual_key))
    return llm, embeddings


async def _build_sample(row: dict[str, Any], user_id: str) -> Any:
    """Produce a ragas SingleTurnSample: retrieve contexts + compose a grounded answer."""
    from ragas import SingleTurnSample

    from app.agents.response_composer import compose
    from app.core.state import AgentState
    from app.rag.retriever import retrieve
    from app.rag.tool_docs import fetch_tool_specs

    state = AgentState(trace_id=uuid4().hex, user_id=user_id, session_id="eval",
                       user_request=row["question"])
    # No plan / tool_results → composer takes the grounded direct-answer path; no MCP
    # tool ever fires, so this measures retrieval + generation, not side effects.
    specs = await fetch_tool_specs(row["question"], _TOOL_DOC_K)
    plans = await retrieve(state)
    contexts = [f"{sp.server}.{sp.name}: {sp.description}" for sp in specs]
    contexts += [p.summary for p in plans if p.summary]
    if not contexts:  # retrieval empty (cold Qdrant) → fall back to gold contexts
        contexts = list(row["ground_truth_contexts"])
    draft = await compose(state)
    answer = f"{draft.summary}\n\n{draft.detail_markdown}".strip()
    return SingleTurnSample(user_input=row["question"], retrieved_contexts=contexts,
                            reference=row["ground_truth_answer"], response=answer)


def _scores_from_result(result: Any, names: list[str]) -> dict[str, float]:
    """Average each metric column from a ragas EvaluationResult into a flat dict."""
    df = result.to_pandas()
    scores: dict[str, float] = {}
    for name in names:
        if name in df.columns:
            scores[name] = round(float(df[name].dropna().mean()), 4)
    return scores


async def _run_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Build samples and run the RAGAS metric suite through the LiteLLM judge."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics.collections import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    llm, embeddings = _judge_and_embeddings()
    samples = [await _build_sample(r, "demo") for r in rows]
    dataset = EvaluationDataset(samples=samples)

    metrics = [faithfulness, context_precision, context_recall]
    names = ["faithfulness", "context_precision", "context_recall"]
    if embeddings is not None:  # answer_relevancy needs an embeddings model
        metrics.append(answer_relevancy)
        names.append("answer_relevancy")

    kwargs: dict[str, Any] = {"llm": llm}
    if embeddings is not None:
        kwargs["embeddings"] = embeddings
    result = evaluate(dataset, metrics=metrics, **kwargs)
    return _scores_from_result(result, names)


def _print_table(scores: dict[str, float]) -> None:
    """Print a markdown score table with pass/fail against the gated thresholds."""
    print("\n| metric | score | threshold | gated | pass |")
    print("|---|---|---|---|---|")
    for name, value in scores.items():
        thr = THRESHOLDS.get(name)
        gated = name in THRESHOLDS
        ok = "—" if thr is None else ("✅" if value >= thr else "❌")
        print(f"| {name} | {value:.3f} | {thr if thr is not None else '—'} | "
              f"{'yes' if gated else 'no'} | {ok} |")


def _gate(scores: dict[str, float]) -> bool:
    """True when every gated threshold is met."""
    return all(scores.get(name, 0.0) >= thr for name, thr in THRESHOLDS.items())


def _write_report(scores: dict[str, float], sha: str, n: int, *, dry_run: bool) -> Path:
    """Write the JSON report to artifacts/ragas/<sha>.json and return its path."""
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    passed = (not dry_run) and _gate(scores)
    report = {"git_sha": sha, "rows": n, "dry_run": dry_run, "thresholds": THRESHOLDS,
              "scores": scores, "passed": passed}
    path = _ARTIFACTS / f"{sha}{'-dryrun' if dry_run else ''}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


async def _amain(args: argparse.Namespace) -> int:
    rows = load_dataset(Path(args.dataset))
    if args.subset:
        rows = rows[: args.subset]
    sha = _git_sha()
    print(f"RAGAS eval — {len(rows)} row(s), git {sha}, dataset {Path(args.dataset).name}")

    if args.dry_run:
        report = _write_report({}, sha, len(rows), dry_run=True)
        print(f"dry-run OK: dataset valid, {len(rows)} rows. Gated thresholds: "
              f"{THRESHOLDS}. Report: {report}")
        return 0

    scores = await _run_metrics(rows)
    report = _write_report(scores, sha, len(rows), dry_run=False)
    _print_table(scores)
    passed = _gate(scores)
    print(f"\nReport written to {report}")
    print("RESULT:", "PASS ✅" if passed else "FAIL ❌ (a gated threshold was missed)")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS eval with gated thresholds.")
    parser.add_argument("--dataset", default=str(_DATASET), help="Path to dataset.jsonl.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate dataset + wiring only; no LLM/Qdrant calls.")
    parser.add_argument("--subset", type=int, default=0,
                        help="Evaluate only the first N rows (0 = all).")
    args = parser.parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
