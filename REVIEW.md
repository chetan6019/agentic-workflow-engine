# Project Review — Agentic Workflow Engine

Reviewed: 2026-06-12. Findings are numbered `R1…R39` for reference.
Severity: 🔴 Critical · 🟠 High · 🟡 Medium · 🟢 Nice-to-have.

Verification baseline: `ruff check` clean · `pytest` 4/4 pass · `mypy --strict` fails on `app/config.py`.

---

## 1. Code Review & Refactoring

### R1 🔴 LiteLLM "primary + fallback" is actually a 50/50 coin flip
**Files:** `infra/litellm_config.yaml`
**Issue:** Each role lists two deployments under the same `model_name` (gpt-oss-120b and gpt-oss-20b) with `routing_strategy: simple-shuffle`. Simple-shuffle load-balances randomly across same-name deployments, so ~half of all planner/composer/judge calls silently go to the 20b model. The "primary, fallback" comment does not match the config's behavior.
**Fix:** One deployment per `model_name`; express degradation through `router_settings.fallbacks` (which already exist):

```yaml
- model_name: planner-default
  litellm_params: { model: groq/openai/gpt-oss-120b, ... }
router_settings:
  fallbacks:
    - planner-default: ["planner-small", "gemini-fallback", "openai-fallback"]
```

### R2 🔴 Planner escalation calls the same model; retries are triplicated
**Files:** `app/orchestration/nodes/planner.py:78-96`, `infra/litellm_config.yaml`
**Issue:** `planner-escalation` resolves to the identical gpt-oss-120b as `planner-default` (only the timeout differs), so `complexity_score > 6` throws away a completed plan and regenerates it with the same model — pure cost/latency waste. The manual try/retry also duplicates LiteLLM's `num_retries: 2` (up to 3×2 attempts).
**Fix:** Remove the manual retry (let the gateway own retries). Either point `planner-escalation` at a genuinely stronger model (e.g. gpt-4o) or delete the escalation branch.

### R3 🟠 Dead code (~200 lines deletable with zero functional loss)
**Files / items:**
- `app/api/invoke.py:84-118` — `_run_and_publish` + `_publish`: never called; duplicates `_run_workflow_with_phase`.
- `app/api/invoke.py:180-208` — SSE endpoint `/v1/invoke/stream/{trace_id}`: nothing publishes during normal runs (UI polls `/phase`); only approval-resume publishes two frames. Also `_publish_resume` in `app/api/approvals.py`.
- `app/data/redis_client.py:22-32` — `set_idempotency_key` / `check_idempotency_key`: unused (MCP client has its own idempotency).
- `app/data/repositories.py:158-162` — `log_action`: never called; the `audit_log` table is a no-op.
- `app/rag/indexer.py:40-49` — `index_preference`: never called; preferences are seed-script-only.
- `app/data/models.py:42-50` — `Message` model: never written or read (candidate for conversation memory — see R12).
- `app/cli.py` — empty file.
**Fix:** Delete each, or wire it (SSE → see R20; `Message` → see R12; `log_action` → approvals/token writes).

### R4 🟠 Duplicated tool-spec retrieval
**Files:** `app/orchestration/nodes/planner.py:23-38` (`_fetch_tool_specs`), `app/agents/response_composer.py:80-101` (`_fetch_tool_catalog`)
**Issue:** Same Qdrant vector-search logic duplicated with different `limit` and error handling.
**Fix:** Extract a single helper, e.g. `app/rag/tool_docs.py::fetch_tool_specs(query: str, k: int) -> list[ToolSpec]` returning `[]` on search failure.

### R5 🟡 `config.py` defeats pydantic-settings and fails mypy --strict
**Files:** `app/config.py`, `app/rag/embedder.py:49,67`
**Issue:** `os.getenv()` field defaults are evaluated once at import, bypass pydantic-settings (`env_file=".env"` already handles env loading), break `mypy --strict` (`str | None` assigned to `str`), and `load_dotenv()` duplicates `env_file`. `embedding_provider` / `hf_embedding_model` are read via `getattr(settings, ...)` but never declared — the `hf` provider path is unreachable.
**Fix:** Plain typed fields; required settings with no default fail loudly:

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str
    embedding_provider: Literal["openai", "hf"] = "openai"
    default_tz: str = "Asia/Kolkata"
```

### R6 🟡 State mutation inside conditional-edge routers
**Files:** `app/orchestration/graph.py:52-62`
**Issue:** `_route_after_guard` mutates `retry_count`, clears the plan, and sets `requires_approval` inside a routing function. LangGraph routers should be pure; these writes are invisible to checkpointing and to readers of the nodes.
**Fix:** Have `guardrails_node` set an explicit `verdict: Literal["finalize","hitl","replan","block"]` on state (including the retry reset); routers only read it.

### R7 🟡 Implicit phase machine via sentinel values
**Files:** `app/orchestration/nodes/orchestrator.py:111-119` (`_phase`), `app/orchestration/graph.py:35`, `app/api/invoke.py:226`
**Issue:** Phase is inferred from which fields are populated, including the fragile `confidence == 0.0` sentinel repeated in three places. One legitimate zero-confidence verdict would mis-route.
**Fix:** Add an explicit `state.phase` field; remove the three inference sites. Aligns with CLAUDE.md "keep state transitions explicit."

### R8 🟢 Smaller cleanups
- `app/api/health.py:30` — `__import__("sqlalchemy", fromlist=["text"]).text(...)` → normal top-of-file import.
- `app/security/jwt_tokens.py:33` — security layer raises `fastapi.HTTPException`; return None/domain error and translate in the router.
- `app/api/auth.py:59-62` — `/me` re-parses the Authorization header the middleware already decoded.
- `app/orchestration/nodes/orchestrator.py:46` — hardcoded `"IST"` label while `default_tz` is configurable.
- `app/rag/retriever.py:100` — stale comment: similarity no longer "feeds the guardrails formula" (guard.py deliberately excludes it).
- `app/mcp/client.py:68-81` — `_resolve` server/tool swap tolerance compensates for planner errors that pre-execution validation (R26) would catch earlier.

---

## 2. Missing Components (ranked by impact)

### R9 🔴 Tests for the paths that can do damage
**Files:** `tests/` (only `test_retriever_hybrid.py` exists)
**Issue:** Approval/resume flow, guard routing bands, MCP idempotency/retry, and `_topo_levels` cycle handling have zero coverage. CLAUDE.md's own minimum list is unmet.
**Fix (minimal):** Three test files reusing the fake-client pattern already established: `test_graph_routing.py` (drive `_route_after_guard` across all four bands), `test_approval_resume.py` (fake repos; assert rebook/reject/edit), `test_mcp_client.py` (fake Redis; assert cache-hit short-circuit and retry-on-timeout).

### R10 🔴 No version control, no CI
**Files:** repo root (no `.git/`, no `.github/workflows/`)
**Issue:** Not a git repository; no pipeline runs ruff/mypy/pytest.
**Fix:** Rotate the `.env` keys first (R11), then `git init` + first commit, plus one GitHub Actions workflow: `ruff check`, `mypy app/`, `pytest`.

### R11 🔴 Live secrets in the working tree
**Files:** `.env`, `Old_code/.env`, `Old_code/client_secret_*.json`
**Issue:** Real, active credentials present: Anthropic/OpenAI/Groq/OpenRouter/Gemini API keys, HF token, Langfuse keys, Google OAuth client secret, and a raw Google access token. `.gitignore` covers `.env`, but nothing is under version control yet, and `docker-compose.yml` mounts the full `.env` into every container (Streamlit and MCP containers receive all LLM provider keys).
**Fix:** Rotate every key now; add a committed `.env.example`; scope per-service environment variables in compose instead of blanket `env_file: .env`.

### R12 🟠 No conversation memory
**Files:** `app/data/models.py:42` (unused `Message`), `streamlit_app/app.py`, `app/prompts.py`
**Issue:** Chat turns live only in Streamlit `session_state`; the planner sees a single utterance, so follow-ups like "move *that* meeting to 4pm" can never resolve.
**Fix (minimal):** Persist user/assistant messages in `/invoke` and on result fetch; pass the last ~6 turns into `build_planner_messages` as a `<context>` block.

### R13 🟠 No run durability / reconciliation
**Files:** `app/api/invoke.py`
**Issue:** Workflows run as in-process asyncio tasks; an API restart mid-run loses the run while the Redis phase key says "running" until TTL.
**Fix (minimal, not a job queue):** Per-run timeout via `asyncio.wait_for` around the graph stream; `/result` reports "failed: instance restarted" when the phase key is gone but the plan row has no terminal signal.

### R14 🟠 No API rate limiting (claimed but absent)
**Files:** `app/main.py` (middleware), README/CLAUDE.md claims
**Issue:** Redis rate limiting is claimed in README and CLAUDE.md but does not exist anywhere in the code.
**Fix (minimal):** Redis `INCR`+`EXPIRE` check (~15 lines) in the JWT middleware. LLM-side, set RPM/TPM limits on LiteLLM virtual keys instead of app code.

### R15 🟡 No auth on trace-read endpoints (IDOR)
**Files:** `app/api/invoke.py:171-227` (`/invoke/phase/`, `/invoke/result/`, `/invoke/stream/`)
**Issue:** No auth and no ownership check — any caller, including unauthenticated ones, can read any run's full final state (emails, calendar contents) given the trace ID. UUIDs are unguessable, but that is security by obscurity.
**Fix:** One ownership check against `plans.user_id` (and require a valid JWT).

### R16 🟡 No DB migration tool
**Files:** `app/data/db.py:43-47`
**Issue:** `create_all` plus a hand-rolled `_MIGRATIONS` string list won't survive the first destructive schema change.
**Fix:** Adopt Alembic; keep `init_db` retry loop for container startup.

### R17 🟡 Broken/contradictory packaging
**Files:** `pyproject.toml`, `requirements.txt`, `README.md`
**Issue:** `pyproject.toml` declares `dependencies = []` and no dev extra, so the README's `uv sync --extra dev` / `pip install -e ".[dev]"` install nothing; `requirements.txt` is fully unpinned; `requires-python = ">=3.13"` contradicts README's "Python 3.12"; README references `infra/docker-compose.yml` (file is at root) and a nonexistent test path.
**Fix:** Move pinned deps into `pyproject.toml` (+ `[project.optional-dependencies].dev`); regenerate `uv.lock`; correct the README.

---

## 3. Deep Dives — Production Readiness

### 3.1 Observability & monitoring

### R18 🔴 App traces and LLM traces never join
**Files:** `app/llm/client.py`, all `ainvoke` call sites
**Issue:** The app's `trace_id` is never sent to LiteLLM, so Langfuse traces cannot be correlated with app logs, runs, or users — "a planner call took 4s" is visible, but not whose run it belonged to.
**Fix:** Pass metadata per call through the gateway (one helper in `client.py`):

```python
await llm.ainvoke(messages, extra_body={"metadata": {
    "trace_id": state.trace_id, "user_id": state.user_id,
    "role": role, "session_id": state.session_id}})
```

LiteLLM forwards `metadata` to Langfuse; every LLM hop becomes attributable end-to-end. Highest-leverage observability change available.

### R19 🟠 No total-run or per-node latency
**Files:** `app/api/invoke.py:143` (`workflow_run_done`), orchestration nodes
**Issue:** Per-tool latency is persisted (`tool_calls.latency_ms`) and per-search timing is logged, but total run duration and per-node durations are not captured anywhere — no p50/p95 derivable for the workflow itself.
**Fix:** Log `duration_ms` in `workflow_run_done`; add a `time.monotonic()` wrapper to the three nodes. Percentiles then come from logs/SQL — no metrics stack needed.

### R20 🟡 Silent quality degradations are logged but not surfaced
**Files:** `app/orchestration/nodes/guard.py:117`, `app/agents/response_composer.py:159`
**Issue:** The judge's neutral fallback verdict and the composer's deterministic fallback produce normal-looking finalized runs; regressions are findable only by grepping logs.
**Fix:** Add `degraded: list[str]` to `AgentState` so the persisted plan row records when a fallback fired.

### 3.2 Cost control

### R21 🔴 LiteLLM response caching is very likely inert
**Files:** `infra/litellm_config.yaml:125-137`
**Issue:** Caching is configured under a top-level `cache:` key, but the LiteLLM proxy expects `litellm_settings.cache: true` + `litellm_settings.cache_params: {...}` (the config's own comment about `callbacks:` notes this exact pitfall class). A per-model-name `ttl:` mapping is also not a supported shape.
**Fix:** Restructure to `litellm_settings: { cache: true, cache_params: { type: redis, host: redis, port: 6379, ttl: 600 } }`; verify with a repeated identical request and a cache-hit in proxy logs.

### R22 🟠 No spend attribution
**Files:** `infra/litellm_config.yaml`, `app/llm/client.py`
**Issue:** One static virtual key (`sk-litellm`) for all traffic lumps all spend together in LiteLLM's tracking DB.
**Fix (gateway-side, no app accounting code):** Pass `user=state.user_id` (LiteLLM tracks per-end-user spend natively) via the same `extra_body` plumbing as R18, and/or issue one virtual key per role so planner vs. judge spend separates in `/spend` reports.

### R23 🟠 Cheap-model routing exists in name only
**Files:** `infra/litellm_config.yaml`
**Issue:** Per-role aliases (`retriever-grader`, `guard-judge`, …) are the right architecture, but all six roles point at the same 120b/20b pair.
**Fix:** Pin low-stakes roles (grader, rewriter, judge — classification/scoring tasks) to gpt-oss-20b deliberately; keep 120b for planner/composer. Config-only change.

### R24 🟡 Token waste
**Files:** `app/rag/retriever.py`, `app/rag/embedder.py`, `infra/litellm_config.yaml`
**Issue:** The same-model escalation re-plan (R2) is the largest waster; `_should_retrieve` spends an LLM call to decide whether to spend more LLM calls (see R25); embeddings are cached twice (app Redis `emb:` keys + proxy cache).
**Fix:** R2 + R25; keep the app-side embedding cache (it works), drop the proxy-side `embed` TTL once R21 is restructured.

### 3.3 Latency

### R25 🔴 Retrieval preamble is 3 sequential LLM round-trips before planning starts
**Files:** `app/rag/retriever.py:191-207`
**Issue:** `_should_retrieve` → `_rewrite_query` → `_grade` are serialized, each a full proxy→provider round-trip, on every request's critical path.
**Fix:** Merge decide+rewrite into one structured call (`{should_retrieve: bool, query: str}` — one schema, one trip); make the grade conditional on candidate count. Removes 1-2 LLM hops per request.

### R26 🟠 Sync Qdrant client blocks the event loop
**Files:** `app/rag/qdrant_client.py:35-39`; call sites in retriever, planner, composer, health
**Issue:** Every `query_points` freezes all concurrent requests for its duration — the main throughput bottleneck under concurrency. Also an explicit CLAUDE.md anti-pattern ("no blocking I/O in async paths").
**Fix:** Swap to `AsyncQdrantClient` — same API surface, mechanical change.

### R27 🟠 Parallelizable steps running sequentially
**Files:** `app/rag/retriever.py:144-161`, `app/agents/response_composer.py:143-147`, `app/orchestration/nodes/planner.py:66`
**Issue:** (a) the fused query and the dense-cosine recovery query run back-to-back; (b) `_fetch_preferences` runs before the no-steps check and tool-catalog fetch; (c) the planner's tool-spec fetch could overlap entry retrieval (both depend only on `user_request`).
**Fix:** `asyncio.gather` for (a) and (b); kick off (c) concurrently in the entry phase.

### R28 🟡 No streaming; SSE machinery dead
**Files:** `app/api/invoke.py`, `streamlit_app/app.py`
**Issue:** The user stares at a phase label until the composer's full draft lands; the SSE endpoint exists but nothing feeds it during normal runs.
**Fix:** Either delete the SSE machinery and accept polling (fits scope), or have `_run_workflow_with_phase` publish state frames to the existing channel and let Streamlit consume it. Token-level streaming through `with_structured_output` is impractical — do not attempt.

### 3.4 RAG evaluation

### R29 🔴 Zero retrieval quality measurement
**Files:** `app/rag/retriever.py:187`, `scripts/`, `feedback` table
**Issue:** No golden set, no precision/recall/MRR; the grader's per-candidate relevance scores are computed then thrown away (used only as a filter); the `feedback` table is collected but never read.
**Fix (in order):** (1) log graded relevance + final ranks per trace; (2) `scripts/eval_retrieval.py` with ~20 hand-labeled (request → expected plan-ids) pairs over seeded data, computing recall@5 and MRR for hybrid vs. dense — the `HYBRID_SEARCH_ENABLED` toggle is the A/B switch and needs no re-indexing; (3) join `feedback` scores to `retrieved_plans` in persisted state for an online signal.

### R30 🟡 Failed-step runs can be indexed as "successful" exemplars
**Files:** `app/rag/indexer.py:33`
**Issue:** `success = state.error is None` — a finalized run with some failed tool steps may be indexed as a successful exemplar and fed to future planners.
**Fix:** Gate on `all(r.ok for r in state.tool_results)` instead.

### 3.5 Hallucination defenses

(Strengths worth keeping: structured outputs everywhere; schema-error vs. transport-error split in `guard.py:33` / `response_composer.py:28`; `_inject_failures` honesty; deterministic approval drafts; pre-execution gates for destructive actions.)

### R31 🔴 The LLM judge cannot block anything, and never sees the evidence
**Files:** `app/orchestration/nodes/guard.py:99,110`, `app/prompts.py` (`build_guard_judge_messages`)
**Issue:** (a) With all tools succeeding, confidence = `0.8 + 0.1·judge + 0.1` ≥ 0.9 even at maximal hallucination risk — the judge is arithmetic decoration; a hallucinated draft with successful tools always finalizes. (b) The judge receives only `draft` + `user_request`, not `tool_results` — it literally cannot detect a fabricated meeting time. The "hallucination_risk" check is not a faithfulness check.
**Fix:** Pass `state.tool_results` into the judge prompt ("is every claim supported by these outputs?") and give the verdict teeth — e.g. cap `confidence = min(computed, 0.84)` when `hallucination_risk > 0.7`. Alternatively, if execution-success-dominates is intentional, delete the judge call and save the LLM hop — but make it honest either way.

### R32 🟠 Plans trusted blindly until execution
**Files:** `app/orchestration/nodes/orchestrator.py`, `app/mcp/client.py:68-81`
**Issue:** Planner output (`tool`/`action`/arguments) is never validated against the known tool registry before execution; the MCP client's `_resolve` swap-tolerance papers over some errors at runtime, others fail mid-run as `unknown_tool`.
**Fix:** Validate each `PlanStep` against the `ToolSpec`s already fetched for the planner prompt; bounce invalid plans straight to re-plan before any tool runs.

### R33 🟡 Retrieved payloads are a prompt-injection path
**Files:** `app/prompts.py`, `app/rag/retriever.py:201` (broadened cross-user search)
**Issue:** Cross-user `plan_json`/`request_text` from Qdrant enter the planner prompt unsanitized — a past user's request text becomes instructions-adjacent content.
**Fix:** Length-clip and tag-escape retrieved text in the prompt builders (one line each).

### R34 🟢 PII patterns scoped narrowly
**Files:** `app/orchestration/nodes/guard.py:38-46`
**Issue:** SSN/credit-card regexes run only on `detail_markdown`, not on `summary`/`actions_*`.
**Fix:** Run `_contains_pii` over all draft text fields.

### 3.6 Memory

### R35 🟠 MemorySaver grows without bound
**Files:** `app/orchestration/graph.py:97`, `app/api/approvals.py:108`
**Issue:** Every run creates a new `thread_id` (= trace_id) checkpoint that is never evicted — a slow leak in a long-lived API process. Resume rebuilds state from Postgres and never reads old checkpoints, so per-thread history is pure overhead.
**Fix:** Delete the thread's checkpoint after terminal persistence (keep MemorySaver for in-flight state only, per CLAUDE.md).

### R36 🟡 Persistence priorities inverted in one spot
**Files:** `app/data/repositories.py:77-85` (`save_plan`), `streamlit_app/app.py`
**Issue:** Full `AgentState` snapshots — including entire tool outputs, which can embed full Gmail bodies — persist forever in `plans.state_json`; meanwhile conversation turns, the thing users expect to persist, live only in Streamlit memory (R12).
**Fix:** Truncate `ToolResult.output` before persistence; implement R12.

(Healthy: all Redis usage is TTL-bounded; `_BG_TASKS` self-cleans; the per-process MCP tool registry in `client.py:63` never refreshes — stale after an MCP redeploy, acceptable at this scale.)

### 3.7 Deployment & scaling

### R37 🟠 In-process run execution is the real horizontal-scaling blocker
**Files:** `app/api/invoke.py:29-33,167`
**Issue:** Because approval-resume rebuilds from Postgres, resumes can land on any instance — the design is more stateless than its README suggests. But runs are asyncio tasks pinned to the accepting instance; kill that pod and the run dies untracked.
**Fix:** R13 (timeout + reconciliation) and document the constraint; a job queue is out of scope.

### R38 🟠 No graceful shutdown; one unguarded fire-and-forget task
**Files:** `app/main.py:21-26`, `app/orchestration/nodes/orchestrator.py:227`
**Issue:** Lifespan has no teardown (no `engine.dispose()`, no Redis close; SIGTERM kills in-flight runs mid-tool-call). `asyncio.create_task(index_plan(...))` holds no reference — the GC-cancellation bug `invoke.py:24-33` correctly guards against elsewhere.
**Fix:** After `yield`, cancel-and-await `_BG_TASKS` with a deadline, then dispose pools; route `index_plan` through the same `_spawn` helper.

### R39 🟡 Compose gaps
**Files:** `docker-compose.yml`
**Issue:** MCP servers and qdrant have no healthchecks and no restart policy (only `api` has one); `api` waits only `service_started` for qdrant/litellm, so first requests can race readiness; all four MCP `env_file: .env` mounts hand every container all provider keys (see R11).
**Fix:** Add healthchecks + `restart: on-failure` to MCP services; scope env vars per service.

(Healthy: SQLAlchemy `pool_pre_ping` + 10/20 pool per instance; local TEI embedding server avoids an external dependency; inbound provider 429 handling is properly centralized in LiteLLM — retries → cooldown → cross-provider fallbacks — once R1 is fixed.)

---

## Top 5 Actions (effort → impact)

1. **R1 + R2 + R21 + R23 — Fix the LiteLLM config** (config-only, ~1 hour). Kill the shuffle, fix or delete escalation, restructure caching, pin cheap models to low-stakes roles. Cost, quality consistency, and caching improve without touching Python.
2. **R18 + R19 + R22 — Join the traces** (~20 lines). Per-call `extra_body` metadata + `user` through the gateway; log run `duration_ms`. End-to-end attribution (run → node → LLM call → spend) falls out of LiteLLM for free.
3. **R31 — Give the guard evidence and teeth** (~40 lines). Feed `tool_results` to the judge and let a bad verdict gate finalization — the gap between "has guardrails" and "guardrails work."
4. **R25 + R26 + R27 — Collapse the retrieval preamble, go async on Qdrant** (~half day). Biggest p50 and throughput win available.
5. **R9 + R10 + R11 — Test the dangerous paths + CI** (~1 day). Rotate keys, `git init`, routing/approval/idempotency tests, 20-line Actions workflow.

**Meta-observation:** the codebase's strength is honest failure handling (schema-vs-transport split, `_inject_failures`, deterministic approval drafts); its weakness is that several *claimed* control systems — HITL band, judge, response cache, rate limits, SSE — are wired to look present but are inert. Each should be made real or explicitly deleted.
