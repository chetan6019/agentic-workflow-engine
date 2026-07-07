# Agentic Workflow Engine

A production-grade AI workflow automation platform. You type a natural-language request into the chat UI (or call the REST API directly); a **LangGraph** state machine plans the work with an LLM, retrieves context from a vector database, calls external tools (**Google** Gmail + Calendar, **GitHub**, **Reddit**, **Finnhub**) through dedicated **MCP servers**, pauses risky steps for **human approval**, and streams the final answer back to your browser.

Every run is fully observable: application logs go to **structlog → Loki → Grafana**, LLM traces are replayable in **Langfuse**, and per-call cost/tokens flow from **LiteLLM → Prometheus → Grafana**.

---

## Architecture (v8)

The canonical diagram is [`ai_workflow_architecture_v8.png`](ai_workflow_architecture_v8.png) at the repo root. In simplified form:

```
┌───────────────────────────────────────────────────────────────────┐
│ 1. UI & API Layer                                                 │
│    React SPA (:8502) ── SSE stream ──┐   Streamlit (:8501, legacy)│
│    FastAPI (:8000) — JWT HS256, rate limits, approvals inbox      │
└──────────────────────────┬────────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────────┐
│ 2. Orchestration Layer — LangGraph state machine                  │
│    intake → plan → retrieve → execute → compose → guard → respond │
│    • Planner: JSON-only plan, self-critique capped at 2           │
│    • Step Guard (PRE-execution): policy.yaml allowlist + regex    │
│      + Redis counters; confidence high|medium|low → HITL pause    │
│    • Response Guard (final): policy / PII check on the answer     │
│    • Checkpoints saved in Postgres (async checkpointer)           │
└───────┬──────────────────┬──────────────────────┬─────────────────┘
        ▼                  ▼                      ▼
┌───────────────┐ ┌──────────────────┐ ┌───────────────────────────┐
│ 3. Data/Cache │ │ 4. Vector DB     │ │ 5. MCP Tool Layer         │
│ PostgreSQL 16 │ │ Qdrant (:6333)   │ │ google  (:7008) Gmail+Cal │
│ (runs, RLS,   │ │ Agentic RAG:     │ │ github  (:7007) OAuth     │
│  approvals)   │ │ route → hybrid   │ │ reddit  (:7006) OAuth     │
│ Redis 7       │ │ search → rerank  │ │ finnhub (:7005) read-only │
│ (cache, rate  │ │ → grade          │ │ tenacity retries +        │
│  limits, idem-│ │                  │ │ idempotency key per call  │
│  potency keys)│ │                  │ │                           │
└───────────────┘ └──────────────────┘ └───────────────────────────┘
        ▼
┌───────────────────────────────────────────────────────────────────┐
│ 6. Response Composer — merges tool outputs into Markdown/JSON     │
│ 7. LLM Gateway & Observability                                    │
│    LiteLLM proxy (:4000): retries, PII redaction, cost → Prometheus│
│    Langfuse: one replayable trace per run                         │
│    Prometheus (:9090) + Grafana (:3000) + Loki (:3100) logs       │
└───────────────────────────────────────────────────────────────────┘
```

### How one request flows (step by step)

1. **You send a request.** The React UI calls `POST /v1/invoke` with your message and JWT. FastAPI validates the token, applies rate limits, and starts a LangGraph run identified by a `trace_id`.
2. **Intake & plan.** The planner LLM (via LiteLLM) decomposes your request into a JSON plan of steps. The output must be valid JSON — on a validation error it retries once, and a self-critique loop (max 2 iterations) can refine the plan.
3. **Retrieve.** The Agentic RAG retriever searches Qdrant (hybrid search + optional rerank) for past workflows, preferences, and tool documentation to ground the plan.
4. **Guard each step (before it runs).** Every planned step carries a `confidence: high|medium|low` tag. The Step Guard checks it against `app/guardrails/policy.yaml` (allowlist/denylist + regex rules + Redis rate counters). Risky steps — anything in `approval_required_actions`, or medium/low confidence — trigger LangGraph's `interrupt()`: the run pauses, an `approvals` row is written to Postgres, and the step waits for a human.
5. **Human approval (HITL).** The approval inbox in the UI shows the pending step. Clicking Approve calls `POST /v1/approvals/{id}`, which resumes the graph with `Command(resume=...)`. Rejecting cancels the step.
6. **Execute tools.** Approved steps run through the MCP client against the four MCP servers. Every tool call carries an idempotency key derived from `(run_id, step_id, tool_input_hash)` (held in Redis for 24h) so retries are always safe. Transient failures retry with tenacity exponential backoff.
7. **Compose & final guard.** The Response Composer merges tool outputs into a readable answer; the Response Guard runs a final policy/PII check on it.
8. **Stream back.** The React UI receives the answer over Server-Sent Events (`GET /v1/invoke/stream/{trace_id}`); Streamlit (legacy) polls `GET /v1/invoke/phase/{trace_id}` and `GET /v1/invoke/result/{trace_id}` instead.
9. **Observe.** The whole run is one Langfuse trace (planner calls, retrieval, tool calls, critique loops); cost and tokens land in Prometheus via LiteLLM; structured JSON logs (with `run_id`, `tenant_id`, `step_id`) flow to Loki and are browsable in Grafana.

---

## Quick Start (local development)

### Prerequisites

- Docker + Docker Compose
- Python 3.12
- Node.js 20+ (only if you work on the React UI)

### 1. Clone and configure

```bash
git clone <repo-url>
cd agentic-workflow-engine
```

Create a `.env` file at the repo root (see [Configuration](#configuration) for the full variable list). Generate the JWT secret:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 2. Start the infrastructure containers

```bash
docker compose up -d postgres redis qdrant litellm-proxy
```

This starts PostgreSQL 16 (:5432), Redis 7 (:6379), Qdrant (:6333), and the LiteLLM proxy (:4000). LiteLLM uses its **own** `litellm` database inside the same Postgres container — never point it at the app's `workflow` database.

### 3. Install Python dependencies

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Optional: local (non-LiteLLM) embeddings — only if you set EMBEDDING_PROVIDER=hf.
# Pulls in torch (~2-3GB); skip this unless you need it.
pip install -r requirements-hf.txt
```

### 4. Apply database migrations and seed data

```bash
alembic upgrade head                    # create all tables
python scripts/seed_demo_data.py        # demo user + sample plans in Qdrant
python scripts/index_tool_docs.py       # index MCP tool docs so the planner can find them
```

> **Important:** the planner only "sees" tools that are indexed in Qdrant. Whenever you add or change an MCP tool, rebuild that server's image and re-run `scripts/index_tool_docs.py`.

### 5. Run the application

```bash
# Backend API
uvicorn app.main:app --reload --port 8000

# React UI dev loop (separate terminal; Vite proxies /v1 → localhost:8000)
cd web && npm install && npm run dev    # opens on :5173

# Legacy Streamlit UI (optional, being retired)
streamlit run streamlit_app/app.py --server.port 8501
```

Log in with the seeded demo user: `demo` / `demo123`.

### 6. Or run everything with Docker Compose

```bash
docker compose build base   # shared dependency image, build this first
docker compose up --build
```

| Service | URL |
|---|---|
| React UI (nginx) | http://localhost:8502 |
| Streamlit (legacy) | http://localhost:8501 |
| FastAPI | http://localhost:8000 (health: `/healthz`, readiness: `/readyz`) |
| LiteLLM proxy | http://localhost:4000 |
| Grafana dashboards | http://localhost:3000 (admin / admin) |
| Prometheus | http://localhost:9090 |

---

## Running in Production

The same compose stack is production-shaped; the steps below are what change when you deploy for real.

1. **Secrets.** Set a strong `JWT_SECRET`, real provider API keys, and OAuth client credentials in the environment (or your secret manager) — never commit `.env`. OAuth tokens are stored as-is in Postgres; rely on disk/at-rest encryption of the database volume.
2. **Build images.** `docker compose build base` first (all Python services derive from it), then `docker compose build`. The four MCP servers share one image (`infra/Dockerfile.mcp`) with different commands.
3. **Migrate before starting the API.** Run `alembic upgrade head` against the production database as a deploy step, then start the services.
4. **Index the tool catalog.** Run `python scripts/index_tool_docs.py` after any MCP server image changes — otherwise the planner will tell users to do things manually because it can't see the tool.
5. **Health checks.** Point your load balancer at `GET /readyz` (checks dependencies) and `GET /healthz` (liveness). Each MCP container has a TCP healthcheck built into compose.
6. **Uvicorn tuning.** Run the API with `--loop uvloop --http httptools` for extra throughput.
7. **Observability.** Prometheus scrapes the API's `/metrics` (RED metrics, guardrail counters, `llm_cost_usd_total`) and the LiteLLM proxy's `/metrics` (spend/tokens). Logs are JSON on stdout → Grafana Alloy → Loki → Grafana. Set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` to get one replayable trace per run.
8. **Multi-tenancy.** Postgres row-level security is enforced with `app.tenant_id` set per session, and Qdrant uses per-tenant collections. Tenant identity always comes from the JWT — never from the request body.
9. **Scaling.** The API is stateless (state lives in Postgres/Redis), so you can run multiple replicas; OAuth token refresh elects a single leader via a Postgres advisory lock, so replicas don't fight.

---

## Configuration

All settings load from `.env` via pydantic-settings (`app/config.py`). Key variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | — | Postgres async connection string (`postgresql+asyncpg://...`) |
| `REDIS_URL` | — | Redis URL (app uses db 0; LiteLLM proxy is isolated in db 1) |
| `QDRANT_URL` | — | Qdrant HTTP endpoint |
| `LITELLM_URL` | — | LiteLLM proxy endpoint (e.g. `http://localhost:4000`) |
| `LITELLM_VIRTUAL_KEY` | — | Virtual key the app sends to the LiteLLM proxy |
| `JWT_SECRET` | — | HS256 signing secret — **must be strong in production** |
| `GROQ_API_KEY` | — | Groq key (default planner/composer/judge models) |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | — | Fallback model providers (see `infra/litellm_config.yaml`) |
| `MCP_GOOGLE_URL` | `http://localhost:7008` | Google MCP server (Gmail + Calendar) |
| `MCP_GITHUB_URL` | `http://localhost:7007` | GitHub MCP server |
| `MCP_REDDIT_URL` | `http://localhost:7006` | Reddit MCP server |
| `MCP_FINNHUB_URL` | `http://localhost:7005` | Finnhub MCP server (read-only) |
| `GOOGLE_OAUTH_CLIENT_ID` / `GOOGLE_OAUTH_CLIENT_SECRET` | — | Google OAuth app credentials |
| `GITHUB_OAUTH_CLIENT_ID` / `GITHUB_OAUTH_CLIENT_SECRET` | — | GitHub OAuth app credentials |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USERNAME` / `REDDIT_PASSWORD` | — | Reddit "script app" credentials |
| `FINNHUB_API_KEY` | — | Finnhub API key |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | optional | Enables Langfuse trace export |
| `EMBEDDING_PROVIDER` | `openai` | Embedding backend; `hf` needs `pip install -r requirements-hf.txt`, `HF_EMBEDDING_MODEL` defaults to `BAAI/bge-base-en-v1.5` |
| `RERANK_ENABLED` | `false` | Toggle retrieval reranking (`Xenova/ms-marco-MiniLM-L-6-v2`) |
| `RATE_LIMIT_PER_MIN` | `60` | Per-user API rate limit |
| `WEB_ORIGINS` | `http://localhost:5173,http://localhost:8502` | Allowed CORS origins for the React UI |

Model routing lives in `infra/litellm_config.yaml`: `planner-default` and `composer` use `groq/openai/gpt-oss-120b`, `retriever-grader` and `guard-judge` use the cheaper `groq/openai/gpt-oss-20b`, with Gemini and OpenAI fallbacks when Groq rate-limits.

---

## Development

```bash
# Lint + type check (CI enforces both)
ruff check app/ streamlit_app/ scripts/ tests/
mypy app/

# Tests — everything is mocked, no containers needed
python -m pytest -q
python -m pytest --cov=app          # CI coverage floor: 60%
python -m pytest tests/test_graph_routing.py -xvs   # single file, verbose

# React UI typecheck + production build
cd web && npm run build

# Migrations
alembic revision --autogenerate -m "describe change"
alembic upgrade head

# RAGAS retrieval-quality evaluation
python scripts/eval/run_ragas.py
```

On Windows, invoke the repo venv explicitly: `.venv\Scripts\python.exe -m pytest`.

---

## Project Layout

```
app/
  main.py          FastAPI entrypoint (uvicorn app.main:app)
  config.py        pydantic-settings config
  prompts.py       ALL prompt builders live here — no inline prompts elsewhere
  api/             Routes: invoke, approvals, auth, sessions, integrations, feedback, health
  orchestration/   graph.py (LangGraph wiring) + nodes/ (orchestrator, planner, guard)
  agents/          response_composer.py
  guardrails/      engine.py, input/output rules, policy.yaml (single source of truth)
  mcp/             client.py (transport map + idempotency) + servers/ (google, github, reddit, finnhub)
  rag/             embedder, indexer, retriever, qdrant_client, tool_docs
  llm/             LiteLLM-backed client factory — the ONLY path to any LLM
  core/            AgentState, Prometheus metrics, background tasks
  data/            async SQLAlchemy models, repositories, Redis client
  security/        JWT, passwords, crypto, log redaction
web/               React SPA (Vite + TypeScript) — talks to the API over HTTP only
streamlit_app/     Legacy Streamlit UI (HTTP only; retired after the React migration)
alembic/           Schema migrations
scripts/           index_tool_docs.py, seed_demo_data.py, eval/run_ragas.py
infra/             Dockerfiles (base/api/mcp/streamlit/web), litellm_config.yaml,
                   prometheus.yml, grafana/, loki/, alloy/
tests/             Flat pytest suite + api/ eval/ guardrails/ mcp/ subfolders (all mocked)
docs/              ADRs + architecture diagram sources
```

---

## Guardrails & Confidence Routing

Guardrails are **pure Python** — allowlist/denylist from `app/guardrails/policy.yaml`, regex rules, and Redis counters. No ML scoring.

| Signal | What happens |
|---|---|
| Step confidence `high`, action allowed | Executes immediately |
| Step confidence `medium` or `low` | Pauses for human approval before executing |
| Action listed in `approval_required_actions` | Always pauses for approval (e.g. sending email, creating events) |
| Denylist / regex rule hit | Blocked with an explanation |
| Trusted read tools (own GitHub repos, Finnhub JSON) | Fast path: auth + rate cap only, no content scanning |

`policy.yaml`'s `approval_required_actions` list is the single source of truth: both the pre-execution gate and the final output check read it. To gate a new write action, add its `tool.action` pair there — no code change needed.

---

## Tech Stack

- **LangGraph** — the only orchestrator; async Postgres checkpointer; native `interrupt()`/`Command(resume=...)` for HITL
- **FastAPI + Uvicorn** — async REST API, JWT HS256 auth, SSE streaming
- **React (Vite + TypeScript)** — chat, run history, approval inbox, integrations; served by nginx
- **LiteLLM** — the only LLM gateway: provider routing, retries, PII redaction, cost/token tracking
- **MCP (four servers)** — google (Gmail + Calendar), github (PyGithub), reddit (script-app OAuth), finnhub (API key, read-only by design)
- **PostgreSQL 16** — system of record with row-level security per tenant; **Redis 7** — cache, rate limits, idempotency keys; **Qdrant** — vectors (per-tenant collections)
- **SQLAlchemy 2.0 async + Alembic** — ORM and migrations
- **Observability, three tools with three jobs:** structlog (app logs → Loki/Grafana), Langfuse (LLM trace replay), LiteLLM → Prometheus (cost/tokens); Grafana dashboards over all of it
