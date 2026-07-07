# 0008 — RAGAS evaluation gates for the RAG layer

- Status: Accepted
- Date: 2026-06-19

## Context

The retrieval + generation path (Qdrant retriever → composer) had only an offline,
ungated recall@k/MRR script (`scripts/eval_retrieval.py`). We want a quantitative,
versioned quality bar so the RAG layer can be refactored with confidence and
regressions are caught in review.

## Decision

Add a RAGAS pipeline (`scripts/eval/run_ragas.py`) over a versioned dataset
(`tests/eval/dataset.jsonl`) with **hard-gated** thresholds:

| Metric | Threshold | Gated |
|---|---|---|
| faithfulness | ≥ 0.90 | yes (runner exits non-zero) |
| context_precision | ≥ 0.80 | yes (runner exits non-zero) |
| context_recall | — | tracked only |
| answer_relevancy | — | tracked only |

- The LLM judge runs **through the LiteLLM proxy** (`langchain_openai.ChatOpenAI`
  pointed at `LITELLM_URL` → `ragas.llms.LangchainLLMWrapper`). No provider SDK is
  imported directly — consistent with the repo's LiteLLM-only rule.
- The eval targets the composer's grounded answer path with **MCP tools mocked**, so
  it measures retrieval + generation, not tool side effects.
- Reports are written to `artifacts/ragas/<git-sha>.json`; a markdown table prints to
  stdout. The runner exits non-zero when a gated threshold is missed.
- CI runs an `eval-ragas` job on PRs touching `app/rag/**`, `app/prompts.py`, or
  `tests/eval/**`. It is **advisory (non-blocking) initially** — CI has no live judge
  or seeded Qdrant yet — mirroring the existing advisory `mypy` step. Flip to blocking
  in a follow-up once a judge model + seeded stack are wired into CI.

## Consequences

- **Changing a gated threshold, or adding/removing dataset rows, requires a new ADR**
  (supersede this one). This keeps the quality bar and its evidence reviewable.
- `ragas` + `datasets` are an optional `eval` extra, kept out of the core app install.
- The thresholds are aspirational until the dataset grows (target 100+ rows) and the
  CI judge is wired; until then the gate is enforced locally / on demand.
