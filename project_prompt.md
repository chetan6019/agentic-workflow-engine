# Project Prompt: AI Workflow Automation Tool (7-Layer, Interview Build)

You are a senior AI engineer and Python developer. Generate **clean, simple, easy-to-understand Python code** for all **7 layers** of the AI Workflow Automation Tool shown in `ai_workflow_architecture_v7_fixed.png`. Code is for an interview build — **readable over clever**, with small modules, flat functions, and single responsibilities.

The system takes a **natural-language user request**, retrieves similar past workflows, decomposes the request into a **typed execution plan**, fans out tool calls across custom **MCP servers** (Calendar, Gmail, Notion, Slack), composes a draft response, runs guardrails to compute a **confidence score**, and either auto-executes or routes to **human-in-the-loop** approval. Every LLM call goes through a **LiteLLM proxy**; all traces go to **Langfuse Cloud**.

Use **OpenAI + Groq** models via LiteLLM; no direct provider SDK usage.
Auth is **username + password → JWT**.

## Core Framework Requirements

- **LangChain** is the primary framework for all agent chains, prompts, embeddings, vector store integration, and output parsing.
- **LangGraph** is the required framework for all stateful multi-step orchestration.
- **Qdrant** is the required vector database for plan retrieval, preference retrieval, and tool capability retrieval.
- **langchain-mcp-adapters** should be used where it helps connect MCP tool schemas or MCP-based tools into LangChain/LangGraph-friendly flows.
- **LiteLLM** is the only allowed gateway for chat and embedding model access.
- Do not bypass these frameworks unless there is a very small, deterministic helper that does not belong inside LangChain or LangGraph.

## Required Libraries

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

Keep the **repository layout small and flat**, with each file focused on one job:

```text
app/
  api/            # FastAPI routers: auth.py, invoke.py, sessions.py, approvals.py, feedback.py, integrations.py, health.py
  orchestration/  # LangGraph graph + nodes (orchestrator, planner, guard)
  agents/         # response_composer.py, retriever.py
  prompts.py      # all prompt builders for every agent (Anthropic-style sections)
  rag/            # qdrant_client.py, retriever.py, indexer.py, embedder.py
  llm/            # client.py (LiteLLM-backed ChatOpenAI), schemas.py
  mcp/
    servers/      # calendar_server.py, gmail_server.py, notion_server.py, slack_server.py
    client.py     # custom async MCP client wrapper
  data/           # db.py (AsyncEngine), models.py, repositories.py, redis_client.py
  security/       # passwords.py (bcrypt), jwt_tokens.py, crypto.py (Fernet)
  logging.py      # structlog setup
  config.py       # Settings (Pydantic BaseSettings)
  main.py         # FastAPI app factory
streamlit_app/
  app.py          # main Streamlit UI
  styles.py       # CSS helpers & theme
infra/
  litellm_config.yaml
  docker-compose.yml
scripts/
  index_tool_docs.py
  seed_demo_data.py
tests/
  # pytest / pytest-asyncio
```

All modules should be short (ideally <150 lines) and easy to explain in 1–2 minutes.

***

## Layer 1 — UI & API (:8501 Streamlit + :8000 FastAPI)

### FastAPI backend

Build a **single FastAPI app** in `app/main.py` with these routers in `app/api/`:

- `auth.py`
  - `POST /v1/auth/register`
  - `POST /v1/auth/login`
  - `GET /v1/auth/me`
- `invoke.py`
  - `POST /v1/invoke` — starts a workflow run (returns `trace_id`)
  - `GET /v1/invoke/stream/{trace_id}` — SSE or chunked JSON streaming of state updates
- `sessions.py`
  - `GET /v1/sessions`
  - `GET /v1/sessions/{id}`
- `approvals.py`
  - `POST /v1/approvals/{token}`
- `feedback.py`
  - `POST /v1/feedback/{trace_id}`
- `integrations.py`
  - `POST /v1/integrations/{provider}/token`
- `health.py`
  - `GET /healthz`
  - `GET /readyz`

**Middleware stack (in this order):**

1. **JWT auth**
   - HS256, 24h tokens, `python-jose`.
   - Reject expired tokens with 401.
   - Attach `user_id` to `request.state.user_id`.
2. **Per-user rate limiting**
   - Redis token bucket via `limits`.
   - 60 req/min, 5000 tokens/min per `user_id`.
3. **CORS**
   - Allow only the Streamlit origin.

Use Pydantic v2 models for all request/response schemas with:

```python
from pydantic import BaseModel, Field, ConfigDict

class InvokeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str | None = Field(default=None)
    user_request: str = Field(..., min_length=4, description="End-user natural language request")
```

Passwords:
- Hash with **bcrypt** (cost 12) in `security/passwords.py`.
- Store `password_hash` in `users` table.

Provider tokens:
- `POST /v1/integrations/{provider}/token` accepts plain token for now.
- Encrypt with **Fernet** (`FERNET_KEY` env var) in `security/crypto.py`.
- Store in `integration_tokens.token_enc` (per user + provider).

### Streamlit frontend

In `streamlit_app/app.py`:

- Theme:
  - `.streamlit/config.toml`:
    ```toml
    [theme]
    base = "dark"
    primaryColor = "#7C3AED"
    font = "Inter"
    ```
  - At app start, inject one `st.markdown` CSS block to:
    - round corners,
    - soften shadows,
    - tighten paddings,
    - style chat bubbles:
      - user (right, violet)
      - assistant (left, slate)

- Layout:
  - `st.set_page_config(layout="wide", page_icon="🤖", page_title="Workflow Agent")`
  - Top-level tabs: `Chat`, `Plan Inspector`, `History`.

- Sidebar:
  - Logo placeholder.
  - Login/Register form (username + password) that talks to `/v1/auth/*`.
  - Connected integrations status as colored pills for Calendar/Gmail/Notion/Slack.
  - Session selector (dropdown) + “New session” button.

- **Chat tab**:
  - Use `st.chat_message` for conversation rendering.
  - Chat input at bottom calls `POST /v1/invoke` and then opens SSE/stream from `/v1/invoke/stream/{trace_id}`.
  - Show per-agent status as inline pills (📥 retrieving → 🧠 planning → 🔧 executing → ✍️ composing → 🛡️ checking).

- **Plan Inspector tab**:
  - Expandable JSON tree of the current `ExecutionPlan`.
  - Confidence gauge.
  - Table of retrieved plans with similarity scores.

- **HITL modal**:
  - When API indicates `requires_approval=True`, show a diff view between original and draft.
  - Buttons: Approve / Edit / Reject, calling `POST /v1/approvals/{token}`.

- **History tab**:
  - List of past sessions from `/v1/sessions` with timestamps and status badges.

Frontend uses **only basic Streamlit primitives**, nicely styled — no heavy front-end frameworks.

***

## Layer 2 — Orchestration Layer (LangGraph + MemorySaver)

Use **LangGraph** to build a single `StateGraph` in `app/orchestration/graph.py` with three nodes:

- `workflow_orchestrator` — entry/exit, talks to data, vector DB, MCP.
- `planner_node` — calls planner LLM to produce a typed `ExecutionPlan`.
- `guardrails_node` — deterministic checks + optional LLM judge → confidence score & routing.

LangChain should be used inside the nodes for:
- prompt construction,
- structured output parsing,
- LLM invocation via LiteLLM aliases,
- message formatting,
- and tool-compatible flows.

Use a Pydantic v2 `AgentState` in `app/core/state.py`:

```python
class AgentState(BaseModel):
    trace_id: str
    user_id: str
    session_id: str
    user_request: str
    retrieved_plans: list[RetrievedPlan] = []
    plan: ExecutionPlan | None = None
    tool_results: list[ToolResult] = []
    draft: DraftResponse | None = None
    confidence: float = 0.0
    requires_approval: bool = False
    approval_token: str | None = None
    retry_count: int = 0
    error: str | None = None
```

Use **LangGraph MemorySaver** for hot state in memory and **Postgres** for long-lived HITL runs:

- In `graph.py`:
  - Wrap the StateGraph with `MemorySaver()` as the default checkpointer.
- For HITL-paused runs, persist `AgentState` snapshot + `trace_id` to Postgres (`plans`/`approvals` tables).

Internal edges:
- Orchestrator → Planner (`plan` label).
- Planner → Orchestrator (`execution plan`).
- Orchestrator → Guardrails (`draft result`).
- Guardrails → Orchestrator (`resume / final`).

Guardrails routing:
- `confidence >= 0.85` → finalize.
- `0.55 <= confidence < 0.85` → `interrupt()` → create `approvals` row with 24h expiry.
- `confidence < 0.55 and retry_count < 2` → increment `retry_count`, go back to Planner.
- `confidence < 0.55 and retry_count == 2` → block with explanation.

Resuming:
- `POST /v1/approvals/{token}` accepts `{decision, edited_draft?}`.
- Loads persisted state, updates `draft` and `requires_approval`, and resumes the graph from the interruption point.

Models:
- Use **gpt-4o-mini** by default.
- Escalate to **gpt-4o** or a **Groq** model (e.g., `groq/llama-3.1-70b`) when planner’s `complexity_score > 6`, all via LiteLLM aliases.

Graph code goes in small files:
- `app/orchestration/nodes/orchestrator.py`
- `app/orchestration/nodes/planner.py`
- `app/orchestration/nodes/guard.py`
- `app/orchestration/graph.py`

***

## Layer 3 — Data & Cache Layer

Implement in `app/data/`:

- `db.py` — async SQLAlchemy engine/session factory (`asyncpg`), Alembic migrations.
- `models.py` — ORM models for tables:
  - `users`
  - `sessions`
  - `messages`
  - `plans`
  - `tool_calls`
  - `approvals`
  - `integration_tokens`
  - `feedback`
  - `audit_log`
- `repositories.py` — simple repository functions per entity.
- `redis_client.py` — single Redis `AsyncRedis` client via `@lru_cache`.

Redis use:
- rate-limit counters
- idempotency keys: `idem:{user_id}:{request_hash}`, TTL 10 min
- LangGraph hot checkpoints if needed beyond MemorySaver
- SSE/pubsub channels for `/v1/invoke/stream/{trace_id}`

No ORM models leak into routers; routers call service/repository functions.

***

## Layer 4 — Vector Database (Agentic RAG with Qdrant)

In `app/rag/`:

- `qdrant_client.py` — create a Qdrant client and collections.
- `embedder.py` — call LiteLLM gateway alias `embed` to get embeddings.
- `retriever.py` — implement the Agentic RAG loop.
- `indexer.py` — background job that indexes new plans & preferences.

Use LangChain for:
- embedding wrappers,
- vector store integration,
- candidate grading structured outputs,
- and output parsing.

Use **Qdrant** as the default and required vector database.

Collections:
- `plans` — past execution plans.
- `preferences` — durable user preferences.
- `tool_capability_docs` — docs + quirks for MCP tools.

Retrieval loop (max 3 iterations):
1. `should_retrieve(state)` — cheap LLM via alias `retriever-grader`.
2. `rewrite_query(user_request)` — alias `retriever-rewriter`.
3. `search(query, filters={user_id, success=True}, k=10)`.
4. `grade(candidates, user_request)` — alias `retriever-grader`, threshold `relevance >= 0.7`.
5. If fewer than 3 remain, broaden filter to `success=True` globally and regrade.
6. Return top-5 as `RetrievedPlan` with similarity.

Embedding and grading always go through the **LLM gateway**.

***

## Layer 5 — MCP Tool Layer

In `app/mcp/servers/`:

- `calendar_server.py` (:7001)
- `gmail_server.py` (:7002)
- `notion_server.py` (:7003)
- `slack_server.py` (:7004)

Each is a focused ~80–120 line module using FastMCP:
- Define Pydantic v2 input models.
- Use `resolve_user_token(user_id, provider)` from `app/mcp/servers/_shared.py`.
- Use `httpx.AsyncClient` to call real provider APIs.
- **No LLM calls** inside servers.

Client:
- `app/mcp/client.py` implements:
  - `class MCPClient`
  - `async def call_tool(server, tool, args) -> ToolResult`
  - retries via `tenacity`
  - Redis-based idempotency key checks
  - `async def list_tools(server)` for discovery

Use **langchain-mcp-adapters** when it meaningfully simplifies adapting MCP tool definitions or schemas into LangChain/LangGraph-friendly flows. Do not use it to hide the custom MCP client/server design; the custom MCP client remains part of the architecture.

Orchestrator execution node:
- fans out parallel steps using `asyncio.gather`
- executes sequential steps in topological order
- records each result in `tool_results` and in `tool_calls` table

***

## Layer 6 — Response Composer

In `app/agents/response_composer.py`:

- Accept:
  - `AgentState.plan`
  - `tool_results`
  - retrieved preferences (top-3)
  - original `user_request`

- Build messages using a function from `app/prompts.py`.
- Call LLM via LiteLLM alias `composer`.
- Parse into `DraftResponse`:

```python
class DraftResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str
    detail_markdown: str
    actions_taken: list[str]
    actions_pending: list[str]
    citations: list[str] = []
```

Rules:
- Inject preferences as `<user_style>` sections in system prompt.
- If any tool result failed, mark it in `actions_pending` with suggested next step; never lie about success.

Return `DraftResponse` to Guardrails node.

***

## Layer 7 — LLM Gateway & Observability (LiteLLM Proxy :4000)

Use **LiteLLM proxy** (configured by `infra/litellm_config.yaml`):

- **Providers**:
  - OpenAI (`gpt-4o-mini`, `gpt-4o`, `text-embedding-3-small`)
  - Groq (e.g. Llama 3.1 70B for heavy planning)
- **Models / aliases**:
  - `planner-default`
  - `planner-escalation`
  - `composer`
  - `retriever-grader`
  - `retriever-rewriter`
  - `guard-judge`
  - `embed`

Follow production practices, simplified:
1. Redis caching (not in-memory) with per-role TTLs.
2. App-side per-user rate limits.
3. Langfuse Cloud callbacks for observability.
4. Pinned model versions.
5. Timeouts + retries in gateway config; app sets `max_retries=0`.
6. Simple regex-based PII masking for logged payload copies only.
7. No K8s/HPA in this prompt; use `docker-compose`.

Application LLM client:

```python
from functools import lru_cache
from langchain_openai import ChatOpenAI
import os

@lru_cache(maxsize=8)
def get_llm(role: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=role,
        openai_api_base=os.environ["LITELLM_URL"],
        openai_api_key=os.environ["LITELLM_VIRTUAL_KEY"],
        timeout=30,
        max_retries=0,
    )
```

***

## Prompt Management — Single `prompts.py`

Use a **single file** `app/prompts.py` that contains pure functions for each agent’s prompts, using Anthropic-style sections with XML-ish tags.

Rules:
- One prompt builder per agent, all in this file.
- Functions are **pure**: no imports of LLM clients, DB, or HTTP.
- All prompts use tagged blocks like `<instructions>`, `<examples>`, `<request>`, `<user_style>`.
- LangChain message lists should be the output shape used by prompt builders.

Suggested prompt builders:
- `build_planner_messages`
- `build_retriever_rewriter_messages`
- `build_retriever_grader_messages`
- `build_guard_judge_messages`
- `build_composer_messages`
- `build_preference_extractor_messages`

***

## General Instructions

- Keep modules **small**, functions **flat**, responsibilities **single**.
- **LangChain is the primary framework for all agent chains, prompts, embeddings, vector store integration, and output parsing.**
- Use **LangGraph StateGraph + MemorySaver** for long-running workflows and interruptions.
- Use **Qdrant** as the required vector database.
- Use **langchain-mcp-adapters** where appropriate for MCP-to-LangChain/LangGraph interoperability.
- Use **Pydantic v2** models everywhere; external schemas use `extra="forbid"`.
- Avoid deep class hierarchies and heavy abstractions.
- Do **not** import provider SDKs directly anywhere in app code.
- Use **gpt-4o-mini** and **Groq** models via LiteLLM; escalate only when needed.
- No K8s, HPA, or complex infra in this prompt; stick to `docker-compose`.
- Every function should be explainable in under 2 minutes during an interview.