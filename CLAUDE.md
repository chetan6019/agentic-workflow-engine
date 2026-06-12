# CLAUDE.md

## Project Overview

This repository contains an interview-grade AI Workflow Automation Tool.

The system accepts a natural-language request, retrieves similar past workflows, creates a typed execution plan, fans out tool calls across custom MCP servers (Calendar, Gmail, Notion, Slack), composes a draft response, runs guardrails with a confidence score, and either auto-completes the workflow or pauses for human approval.

The main engineering goal is readability over cleverness. Code should be easy to explain in an interview, with small modules, flat functions, and clear boundaries between API, orchestration, retrieval, integrations, and persistence.

## Product Flow

1. User sends a request from Streamlit UI.
2. FastAPI receives the request and starts a workflow run.
3. LangGraph orchestrator loads state and decides next actions.
4. Retriever searches similar plans and user preferences from Qdrant.
5. Planner creates a typed `ExecutionPlan`.
6. Orchestrator executes plan steps through MCP servers.
7. Response composer creates a `DraftResponse`.
8. Guardrails compute confidence and choose one of: finalize, re-plan, or interrupt for human approval.

## Architecture Layers

### Layer 1 — UI & API
- Streamlit frontend on `:8501`
- FastAPI backend on `:8000`
- JWT auth, rate limits, sessions, approvals, feedback, integrations

### Layer 2 — Orchestration
- LangGraph `StateGraph`
- Nodes:
  - `workflow_orchestrator`
  - `planner_node`
  - `guardrails_node`
- `MemorySaver` for hot state/checkpointing

### Layer 3 — Data & Cache
- PostgreSQL is the system of record
- Redis for rate limits, idempotency, cache, pub/sub, hot workflow state
- SQLAlchemy 2.0 async + asyncpg

### Layer 4 — Vector Retrieval
- Qdrant stores:
  - past plans
  - user preferences
  - tool capability documents
- Retrieval is agentic, not a single vector search

### Layer 5 — MCP Tool Layer
- Custom MCP servers for Calendar, Gmail, Notion, Slack
- Custom MCP client used by orchestrator
- Deterministic tools only; no LLM calls inside MCP servers

### Layer 6 — Response Composer
- Converts plan + tool results + preferences into user-facing draft output
- Must surface partial failures honestly

### Layer 7 — LLM Gateway & Observability
- LiteLLM proxy is the only gateway for chat and embedding models
- Langfuse Cloud receives LLM traces
- App logs go through structlog to stdout

## Core Framework Rules

- **LangChain** is the primary framework for all agent chains, prompts, embeddings, vector store integration, and output parsing.
- **LangGraph** is the required framework for all stateful multi-step orchestration.
- **Qdrant** is the required vector database for plan retrieval, preference retrieval, and tool capability retrieval.
- **langchain-mcp-adapters** should be used where it helps connect MCP tool schemas or MCP-based tools into LangChain/LangGraph-friendly flows.
- **LiteLLM** is the only allowed gateway for chat and embedding model access.
- Do not bypass these frameworks unless there is a very small, deterministic helper that does not belong inside LangChain or LangGraph.

## Tech Stack

Primary libraries:
- `langchain`
- `langgraph`
- `langchain-openai`
- `langchain-mcp-adapters`
- `qdrant-client`
- `fastapi`
- `streamlit`
- `sqlalchemy`
- `asyncpg`
- `redis`
- `httpx`
- `pydantic`
- `python-jose`
- `bcrypt`
- `cryptography`

## Mandatory Engineering Rules

### Code Style
- Prefer readability over cleverness.
- Keep modules small and focused.
- Prefer flat functions over inheritance-heavy designs.
- Avoid deep abstraction unless it removes real duplication.
- Every file should be understandable in about 30 seconds.
- Every important function should have a short docstring.

### Python Rules
- Use async-first code for all I/O paths.
- Use `httpx.AsyncClient`, not `requests`.
- Use SQLAlchemy 2.0 async APIs.
- Use Pydantic v2 for all request, response, and shared schemas.
- External-facing schemas should use `ConfigDict(extra="forbid")`.
- Internal state models may use more relaxed config when needed.

### LangChain / LangGraph Rules
- LangChain is the primary framework for prompts, message construction, structured outputs, embeddings, vector store integration, and output parsing.
- LangGraph owns all stateful orchestration, routing, retries, and HITL interrupt/resume behavior.
- Use `langchain-mcp-adapters` where it meaningfully simplifies MCP interoperability.
- Do not scatter prompt strings in business logic.
- Keep prompt builders centralized and pure.

### LLM Rules
- All LLM and embedding calls must go through LiteLLM.
- Never import provider SDKs directly in app code.
- Do not call OpenAI, Anthropic, or Groq SDKs directly.
- Use `app/llm/client.py` as the only LLM construction entry point.

### Prompt Rules
- Keep prompt builders centralized.
- Use a single `app/prompts.py` file.
- Prompt builders must be pure functions.
- Prompt builders return message lists only.
- No I/O inside prompt builders.
- Prefer Anthropic-style tagged sections such as:
  - `<instructions>`
  - `<context>`
  - `<examples>`
  - `<request>`
  - `<output_format>`
  - `<user_style>`

### Orchestration Rules
- Multi-step workflow logic belongs in LangGraph.
- Do not implement workflow state transitions ad hoc in routers.
- Keep state transitions explicit.
- Retry and approval logic must be visible in graph code.
- Use `MemorySaver` for hot workflow state.
- Persist approval-related state in Postgres for resumable HITL runs.

### MCP Rules
- MCP servers are deterministic adapters only.
- No LLM calls inside MCP servers.
- MCP server files should stay small and focused.
- The orchestrator must call tools through the custom MCP client.
- Handle idempotency in the client, not individually inside every tool.
- `langchain-mcp-adapters` can help with integration, but should not hide the custom MCP client/server architecture.

## Repository Layout

```text
app/
  api/            # FastAPI routers
  orchestration/  # LangGraph graph and nodes
  agents/         # Composer and helper agents outside graph nodes
  rag/            # Qdrant retrieval, embeddings, indexing
  llm/            # LiteLLM-backed client and LLM schemas/helpers
  mcp/            # MCP servers and client
  data/           # DB engine, ORM models, repositories, Redis
  security/       # bcrypt, JWT, Fernet helpers
  core/           # shared state, config helpers, errors
  logging.py      # structlog setup
  config.py       # app settings
  main.py         # FastAPI app entrypoint
streamlit_app/
  app.py          # Streamlit UI
  styles.py       # Streamlit CSS helpers
infra/
  litellm_config.yaml
  docker-compose.yml
scripts/
  index_tool_docs.py
  seed_demo_data.py
tests/
```

## File Placement Rules

- New API endpoints go under `app/api/`.
- New shared schemas go under `app/core/` or `app/llm/` depending on ownership.
- New LangGraph node logic goes under `app/orchestration/nodes/`.
- Graph wiring belongs in `app/orchestration/graph.py`.
- New prompt builders belong in `app/prompts.py`.
- New retrieval logic belongs in `app/rag/`.
- Qdrant-specific integration code belongs in `app/rag/`.
- New integration adapters belong in `app/mcp/servers/`.
- New DB access logic belongs in `app/data/repositories.py` or small focused repository modules.
- New security helpers belong in `app/security/`.
- Do not place business logic inside `main.py`.

## API Surface

Expected API routes:
- `POST /v1/auth/register`
- `POST /v1/auth/login`
- `GET /v1/auth/me`
- `POST /v1/invoke`
- `GET /v1/invoke/stream/{trace_id}`
- `GET /v1/sessions`
- `GET /v1/sessions/{id}`
- `POST /v1/approvals/{token}`
- `POST /v1/feedback/{trace_id}`
- `POST /v1/integrations/{provider}/token`
- `GET /healthz`
- `GET /readyz`

Keep routes thin. Routers should validate, delegate, and return typed responses.

## LangGraph Rules

`AgentState` is the central shared state. It should remain explicit and typed.

Core fields include:
- `trace_id`
- `user_id`
- `session_id`
- `user_request`
- `retrieved_plans`
- `plan`
- `tool_results`
- `draft`
- `confidence`
- `requires_approval`
- `approval_token`
- `retry_count`
- `error`

Guardrails routing rules:
- `confidence >= 0.85` → finalize
- `0.55 <= confidence < 0.85` → interrupt for HITL approval
- `confidence < 0.55 and retry_count < 2` → re-plan
- `confidence < 0.55 and retry_count == 2` → block with explanation

Do not hide these thresholds in random helper files.

## Data Rules

### PostgreSQL
System of record tables include:
- `users`
- `sessions`
- `messages`
- `plans`
- `tool_calls`
- `approvals`
- `integration_tokens`
- `feedback`
- `audit_log`

### Redis
Use Redis for:
- rate limiting
- idempotency keys
- hot workflow state
- short TTL caches
- SSE/pub-sub

### Qdrant
Collections include:
- `plans`
- `preferences`
- `tool_capability_docs`

The retriever should:
- decide if retrieval is needed,
- rewrite queries,
- search user-scoped plans first,
- broaden to global successful plans if needed,
- grade candidates,
- return top relevant items.

## Security Rules

- Passwords must be hashed with bcrypt.
- JWT uses HS256 with 24h expiry.
- Provider tokens must be Fernet-encrypted before storing in Postgres.
- Never log raw secrets, tokens, passwords, or decrypted credentials.
- Avoid exposing internal error traces in API responses.

## Frontend Rules

The Streamlit app is the first thing an interviewer sees.

UI rules:
- dark theme
- attractive but simple
- Streamlit primitives only
- custom CSS at startup
- clear chat history
- plan inspector tab
- history tab
- visible approval flow
- clean status pills for workflow phases

Do not over-engineer the frontend. It should look polished, not fancy.

## Testing Rules

Use:
- `pytest`
- `pytest-asyncio`
- Testcontainers for Postgres, Redis, and Qdrant where useful
- `respx` for mocking LiteLLM HTTP calls if needed

Minimum expectations:
- tests for auth flows
- tests for repositories
- tests for orchestrator routing behavior
- tests for approval resume behavior
- tests for retrieval grading logic
- tests for MCP client idempotency and retries

## CI Expectations

Code should be compatible with:
- `ruff`
- `mypy --strict`
- `pytest`
- docker-based integration checks

Do not introduce patterns that make static analysis unnecessarily hard.

## Preferred Model Usage in Claude Code

### Prefer Sonnet for
- FastAPI routers
- Streamlit UI
- SQLAlchemy models
- repositories
- security helpers
- config files
- simple MCP servers
- straightforward tests
- simple composer wiring

### Prefer Opus for
- LangGraph graph design
- orchestrator node
- planner node
- guardrails node
- invoke/approval flow
- shared state design
- retrieval loop design
- MCP client behavior
- prompt architecture
- cross-file refactors
- subtle async/state/debugging issues

Simple rule:
- Sonnet writes individual files.
- Opus coordinates difficult files that must work together.

## Editing Guidance for Claude

When making changes:
1. Read the surrounding files first.
2. Prefer minimal, targeted edits.
3. Preserve existing architecture unless explicitly asked to change it.
4. Do not rename or move files without a strong reason.
5. Do not introduce a new framework when existing tools already solve the problem.
6. If a design choice is unclear, choose the simplest implementation consistent with this file.

## Anti-Patterns to Avoid

Do not:
- add giant service classes for small tasks
- hide control flow behind decorators or metaprogramming
- spread prompts across multiple unrelated modules
- put orchestration logic in FastAPI routers
- put provider-specific LLM code all over the codebase
- use blocking I/O in async paths
- fabricate successful tool results when a tool call failed
- add K8s-specific complexity into the interview build unless explicitly requested

## Definition of Done

A change is considered done when:
- code is readable and small,
- types are clear,
- modules follow the intended layer boundaries,
- tests cover the main success and failure paths,
- logs are useful but safe,
- partial failures are surfaced honestly,
- the result is easy to explain in an interview.