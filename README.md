# Agentic Workflow Engine

An interview-grade AI workflow automation platform. A user types a natural-language request into a Streamlit chat (or hits the FastAPI endpoint); a LangGraph state machine plans the work with an LLM, retrieves context from Qdrant, executes tool calls through custom MCP servers (Calendar, Gmail, Notion, Slack), gates risky steps behind a human approval flow, and returns the answer while the UI polls live phase progress.

---

## Architecture

```
User (Streamlit :8501)
    │
    ▼
FastAPI (:8000)   ← JWT auth, sessions, approvals, phase polling
    │
    ▼
LangGraph StateGraph
    ├── retriever  → Qdrant (past plans + preferences + tool docs)
    ├── planner    → LiteLLM (gpt-4o-mini / llama-3.1-70b)
    ├── orchestrator → MCP Client → Calendar | Gmail | Notion | Slack
    ├── composer   → LiteLLM (gpt-4o-mini)
    └── guardrails → confidence score → finalize | HITL | re-plan | block
    │
    ▼
PostgreSQL (system of record)  Redis (cache, idempotency, hot state, phase)
LiteLLM proxy (:4000)          Qdrant (:6333)
Langfuse (LLM trace viewer)
```

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Python 3.13
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### 1. Clone and configure

```bash
git clone <repo-url>
cd agentic-workflow-engine
cp .env .env.local   # then fill in OPENAI_API_KEY, JWT_SECRET, FERNET_KEY
```

Generate the required secrets:

```bash
# JWT secret
python -c "import secrets; print(secrets.token_hex(32))"

# Fernet key for encrypting provider tokens
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 2. Start infrastructure

```bash
docker compose up -d postgres redis qdrant litellm-proxy
```

### 3. Install Python dependencies

```bash
uv sync --extra dev   # creates .venv and installs everything
# or: pip install -e ".[dev]"
```

### 4. Initialize the database and seed demo data

```bash
uv run python -c "import asyncio; from app.data.db import init_db; asyncio.run(init_db())"
uv run python scripts/seed_demo_data.py
uv run python scripts/index_tool_docs.py
```

### 5. Run the services

```bash
# FastAPI backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Streamlit frontend (separate terminal)
uv run streamlit run streamlit_app/app.py --server.port 8501
```

Open `http://localhost:8501` — log in with `demo` / `demo123`.

### 6. Run everything with Docker Compose

```bash
docker compose up --build
```

---

## Configuration

All settings are read from `.env` (or environment variables). Key variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` | Postgres async connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint |
| `LITELLM_URL` | `http://localhost:4000` | LiteLLM proxy endpoint |
| `LITELLM_VIRTUAL_KEY` | `sk-litellm` | Virtual key for LiteLLM |
| `OPENAI_API_KEY` | — | OpenAI API key (forwarded by LiteLLM) |
| `GROQ_API_KEY` | — | Groq API key (used for escalation model) |
| `JWT_SECRET` | `change-me` | HS256 signing secret — **must change** |
| `FERNET_KEY` | — | Fernet key for encrypting OAuth tokens |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse project public key (optional) |
| `LANGFUSE_SECRET_KEY` | — | Langfuse project secret key (optional) |

---

## Development

```bash
# Lint
uv run ruff check app/ streamlit_app/

# Type check (strictest file)
uv run mypy app/core/state.py --strict

# Tests
uv run pytest

# Run a single test
uv run pytest tests/test_graph_routing.py -xvs
```

---

## Project Layout

```
app/
  api/            FastAPI routers (auth, invoke, approvals, sessions, feedback, integrations, health)
  orchestration/  LangGraph graph + planner / orchestrator / guardrails nodes
  agents/         Response composer
  rag/            Qdrant retrieval, embeddings, indexing
  llm/            LiteLLM-backed client and PII redaction
  mcp/            MCP client (langchain-mcp-adapters) + 4 MCP servers
  data/           SQLAlchemy models, DB engine, Redis, repositories
  security/       bcrypt, JWT, Fernet helpers
  core/           Shared Pydantic state models
  logging.py      structlog JSON configuration
  config.py       Pydantic-Settings app config
  main.py         FastAPI app factory
streamlit_app/
  app.py          Streamlit UI (chat, plan inspector, history, HITL)
  styles.py       CSS injection helpers
docker-compose.yml       Full service stack (run from the repo root)
infra/
  litellm_config.yaml    LiteLLM model aliases + caching config
  Dockerfile.base        Shared dependency image
  Dockerfile.api         FastAPI container
  Dockerfile.mcp         MCP server container
  Dockerfile.streamlit   Streamlit container
alembic/                 Schema migrations (alembic upgrade head)
scripts/
  seed_demo_data.py      Creates demo user + 20 seed plans in Qdrant
  index_tool_docs.py     Indexes MCP tool docstrings into Qdrant
tests/                   Flat pytest suite (routing, approval, guard, MCP, retrieval…)
```

---

## Guardrails Routing

| Confidence | Action |
|---|---|
| `≥ 0.85` | Auto-finalize |
| `0.55 – 0.85` | Interrupt for HITL approval |
| `< 0.55`, retries left | Re-plan |
| `< 0.55`, no retries | Block with explanation |

---

## Tech Stack

- **LangGraph** — stateful multi-step orchestration with `interrupt()` / HITL resume
- **LangChain** — prompts, structured outputs, embeddings, vector store integration
- **LiteLLM** — unified LLM gateway (OpenAI + Groq); only allowed model access path
- **FastMCP** — MCP server implementation (Calendar, Gmail, Notion, Slack)
- **langchain-mcp-adapters** — MCP client connectivity into LangChain/LangGraph
- **Qdrant** — vector database for plans, preferences, tool capability docs
- **FastAPI** — async REST API with JWT auth, sessions, approvals; UI polls live phase
- **Streamlit** — dark-themed chat UI with approval inbox and plan inspector
- **PostgreSQL** — system of record (users, sessions, plans, approvals, tokens)
- **Redis** — idempotency keys, embedding cache, hot workflow state, phase polling
- **structlog** — structured JSON logging; **Langfuse** — LLM trace visualization
