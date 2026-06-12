# Claude Code Prompt — Opus 4.7
# Role: Design and implement the architecture-critical, cross-cutting, and orchestration-heavy files.
# Run this FIRST, before the Sonnet 4.6 prompt.
# The files you create here are the foundation everything else builds on.

---

## Read First

Before writing a single line of code:
1. Read `project_prompt.md` — complete architecture reference.
2. Read `CLAUDE.md` — project rules, non-negotiables, and working style.
3. Read `ai_workflow_architecture_v7_fixed.png` — study the diagram until every edge is clear.

Confirm you have read all three before proceeding.

---

## Framework Mandate

The latest project rules are non-negotiable:

- **LangChain** is the primary framework for all agent chains, prompts, embeddings, vector store integration, structured outputs, and output parsing.
- **LangGraph** is the required framework for all stateful multi-step orchestration.
- **Qdrant** is the required vector database for plan retrieval, preference retrieval, and tool capability retrieval.
- **langchain-mcp-adapters** should be used where it meaningfully improves MCP-to-LangChain/LangGraph interoperability.
- **LiteLLM** is the only allowed gateway for chat and embedding model access.

Design every public interface with those constraints in mind.

---

## Your Job

You are creating the **10 hardest files** in this project:
- The shared Pydantic state models that every other file imports.
- The LangGraph orchestration graph and all three nodes.
- The Agentic RAG retriever loop.
- The custom MCP client.
- The two API endpoints that coordinate the entire workflow.
- The full Streamlit UI.

Everything you build will be imported by the Sonnet prompt's downstream files. **Get the interfaces right.** If a function signature or model field needs to change later, every downstream file breaks.

---

## Constraints (do not violate)

- Every file must be under 150 lines. Split into sub-modules or functions in the same file if needed.
- No imports of `openai`, `anthropic`, or any provider SDK. Use `app/llm/client.py:get_llm(role)` only.
- Use **LangChain** primitives for message construction, structured output parsing, model invocation, and vector-store-facing code.
- Use **LangGraph** for workflow state, routing, retries, and HITL pause/resume.
- Use **Qdrant** as the vector database; do not substitute another vector store.
- Use `langchain-mcp-adapters` where it simplifies tool/schema interoperability, but do not hide the custom MCP client/server architecture.
- No Claude models. All model access must go through LiteLLM aliases via `get_llm(role)`. Support multiple providers per role, with provider selection handled by LiteLLM routing, fallback, load balancing, health checks, or config-driven policy not by a hardcoded `complexity_score` threshold in application code.
- Pydantic v2 everywhere. `model_config = ConfigDict(extra="forbid")` on external schemas.
- Async-first throughout. `asyncpg`, `httpx.AsyncClient`. Never `requests`.
- Use `@lru_cache` for singletons (Qdrant client, Redis, AsyncEngine, LLM clients).
- Every function must have a 1–2 line meaningful docstring related to it functionality.
- Use LangGraph `MemorySaver` for hot state; persist to Postgres for HITL-paused runs only.

---

## Files to Create

Create these files in this order because later files depend on earlier ones.

### 1. `app/core/state.py` — All shared Pydantic models

Create this file first. Everything else imports from it.

Define all of these models, fully typed with `Field(description="...")` on every field:

```python
class ToolSpec(BaseModel):          # name, description, server, input_schema
class PlanStep(BaseModel):          # id, tool, action, arguments, depends_on: list[str]
class ExecutionPlan(BaseModel):     # reasoning, steps, strategy (sequential/parallel/mixed),
                                    # complexity_score: int (1-10), estimated_cost_usd, requires_approval
class ToolResult(BaseModel):        # step_id, ok: bool, output: dict|None, error: str|None, latency_ms
class RetrievedPlan(BaseModel):     # plan_id, request_text, plan_json, summary, similarity: float
class DraftResponse(BaseModel):     # summary, detail_markdown, actions_taken, actions_pending, citations
class AgentState(BaseModel):        # trace_id, user_id, session_id, user_request,
                                    # retrieved_plans, plan, tool_results, draft,
                                    # confidence, requires_approval, approval_token,
                                    # retry_count, error
```

Rules:
- All models use `model_config = ConfigDict(extra="forbid")`.
- `AgentState` uses `extra="ignore"` instead for LangGraph forward-compatibility.
- Export everything cleanly so imports are `from app.core.state import AgentState`.

### 2. `app/orchestration/nodes/planner.py` — Planner Node

- `async def planner_node(state: AgentState) -> AgentState`
- Fetches `tool_specs` from Qdrant `tool_capability_docs` collection via `app/rag/qdrant_client`.
- Uses LangChain-compatible prompt messages from `build_planner_messages(...)` in `app/prompts.py`.
- Calls `get_llm("planner-default").with_structured_output(ExecutionPlan).ainvoke(messages)`.
- If `plan.complexity_score > 6`, re-calls with `get_llm("planner-escalation")`.
- If `with_structured_output` raises twice, sets `state.error` and returns.
- Returns updated `AgentState` with `plan` set.
- No side effects beyond state mutation.

### 3. `app/orchestration/nodes/guard.py` — Guardrails Node

- `async def guardrails_node(state: AgentState) -> AgentState`

**Stage 1 — deterministic (no LLM):**
- PII regex check on outgoing `state.draft.detail_markdown` — block if found.
- Destructive action check: if any `plan.step.action` is in `{"delete_event", "delete_page"}` → set `requires_approval=True`.
- Scope check: if any step uses a tool the user has no `integration_tokens` row for → block.

**Stage 2 — LLM judge (only if Stage 1 passes):**
- Calls `build_guard_judge_messages(draft, user_request)` from `app/prompts.py`.
- Calls `get_llm("guard-judge").with_structured_output(GuardVerdict).ainvoke(messages)`.

`GuardVerdict` fields:
```python
tone_fit: float
hallucination_risk: float
instruction_adherence: float
```

Confidence score:
```python
confidence = (
    0.4 * tool_success_rate
    + 0.3 * max_retrieval_similarity
    + 0.2 * llm_judge_avg
    + 0.1 * schema_ok
)
```

Store `state.confidence` and `state.requires_approval`.

### 4. `app/orchestration/nodes/orchestrator.py` — Workflow Orchestrator Node

- `async def orchestrator_node(state: AgentState) -> AgentState`
- This node runs multiple times, so inspect `state` to decide the phase.

Responsibilities:
- On entry: load session history from Postgres, call `app/rag/retriever.retrieve(state)`, store results in `state.retrieved_plans`.
- Post-plan: fan out MCP tool calls using `asyncio.gather` for parallel steps; sequential steps in DAG order; record each in `state.tool_results` and in the `tool_calls` DB table.
- Post-compose: prepare state for the guard node via graph edge; do not call guard directly.
- On finalize: persist plan to Postgres, enqueue `indexer.index_plan(state)` as a background task.

Imports:
- `MCPClient` from `app/mcp/client.py`
- `retrieve` from `app/rag/retriever.py`

### 5. `app/orchestration/graph.py` — LangGraph Graph Wiring

- Build a `StateGraph(AgentState)` with nodes: `"orchestrator"`, `"planner"`, `"guardrails"`.
- Entry point: `"orchestrator"`.

Edges:
- `"orchestrator"` → `"planner"` on first pass.
- `"planner"` → `"orchestrator"` after plan creation.
- `"orchestrator"` → `"guardrails"` after compose phase.
- `"guardrails"` → conditional:
  - `confidence >= 0.85` → `END`
  - `0.55 <= confidence < 0.85` → `interrupt()` → `END`
  - `confidence < 0.55 and retry_count < 2` → `"planner"`
  - else → `END` with `state.error = "low_confidence_blocked"`

Rules:
- Use `MemorySaver()` as the default checkpointer.
- Expose `compile_graph() -> CompiledGraph` as an `@lru_cache` singleton.
- Keep wiring small; node logic belongs in node files only.

### 6. `app/rag/retriever.py` — Agentic RAG Loop

- `async def retrieve(state: AgentState) -> list[RetrievedPlan]`

Implements the loop:
1. `_should_retrieve(state)` — calls `get_llm("retriever-grader")`, returns `bool`
2. `_rewrite_query(user_request)` — calls `get_llm("retriever-rewriter")`
3. `_search(query, filters, k=10)` — Qdrant cosine search on `plans`
4. `_grade(candidates, user_request)` — calls `get_llm("retriever-grader")`, drops hits below `relevance=0.7`
5. If fewer than 3 remain, re-search without user scope and re-grade
6. Return top-5 as `list[RetrievedPlan]`

Rules:
- Each LLM call uses `build_retriever_*_messages(...)` from `app/prompts.py`.
- All embeddings go through `app/rag/embedder.embed_text`.
- Use LangChain-compatible vector integration patterns around Qdrant where appropriate.

### 7. `app/mcp/client.py` — MCP Client via `langchain-mcp-adapters`

- Use `langchain-mcp-adapters` as the only implementation library for MCP client connectivity.
- `class MCPClient`
  - `__init__(self, server_config: dict[str, dict])`
  - wraps `MultiServerMCPClient` from `langchain_mcp_adapters.client`
  - maintains a stable project-specific interface for orchestration code
- `async def call_tool(self, server: str, tool: str, args: dict) -> ToolResult`
- `async def list_tools(self, server: str) -> list[ToolSpec]`

`call_tool` behavior:
- Compute `idempotency_key = sha256(f"{args.get('trace_id', '')}:{tool}")`
- Check Redis for cached result; return if found
- Resolve the requested tool from the `MultiServerMCPClient` tool registry
- Invoke the tool through `langchain-mcp-adapters`
- Retry with `tenacity` on transient failures only
- Store normalized result in Redis with TTL 15 min
- Return `ToolResult`

`list_tools` behavior:
- Load available tools from `MultiServerMCPClient`
- Filter tools for the requested server
- Convert each discovered tool into the project’s `ToolSpec`

Rules:
- `get_mcp_client() -> MCPClient` is an `@lru_cache` singleton reading MCP server config from `config.py`
- Do not implement direct MCP HTTP calls manually in this file
- Do not use the `mcp` Python SDK directly in this file
- Do not POST directly to `{base_urls[server]}/tools/{tool}`
- `langchain-mcp-adapters` is the source of truth for MCP connectivity and tool discovery
- Keep the thin `MCPClient` wrapper so orchestrator code depends only on project-local interfaces

### 8. `app/api/invoke.py` — POST /v1/invoke + GET /v1/invoke/stream/{trace_id}`

`POST /v1/invoke`:
- Accepts `InvokeRequest(session_id, user_request)`
- Creates or resumes session in Postgres
- Mints `trace_id` via UUID4
- Builds initial `AgentState`
- Enqueues graph run:
  ```python
  compile_graph().ainvoke(state, config={"configurable": {"thread_id": trace_id}})
  ```
- Publishes state updates to Redis pubsub channel `sse:{trace_id}`
- Returns `{"trace_id": trace_id}` immediately

`GET /v1/invoke/stream/{trace_id}`:
- SSE endpoint using `StreamingResponse`
- Subscribes to `sse:{trace_id}`
- Yields `data: {json}\n\n`
- Sends final `data: {"done": true}\n\n` on completion
- Closes cleanly on client disconnect

### 9. `app/api/approvals.py` — POST /v1/approvals/{token}

- Accepts `ApprovalRequest(decision: Literal["approve","edit","reject"], edited_draft: DraftResponse | None)`
- Load `approvals` row by token; raise `404` if missing, `410` if expired
- Load persisted `AgentState` snapshot from `plans` table by `trace_id`

Decision handling:
- `"approve"` → set `state.requires_approval = False`
- `"edit"` → replace `state.draft` with `edited_draft`, set `requires_approval = False`
- `"reject"` → set `state.error = "rejected_by_user"` and return early

Then:
```python
compile_graph().ainvoke(state, config={"configurable": {"thread_id": state.trace_id}})
```

- Publish resume event to `sse:{trace_id}`
- Return `{"status": "resumed"}`

### 10. `streamlit_app/app.py` — Full Streamlit UI

Build the full UI. Import and call `inject_styles()` from `streamlit_app/styles.py` first.

Config:
```python
st.set_page_config(layout="wide", page_icon="🤖", page_title="Workflow Agent")
```

Sidebar:
- Title: `"### 🤖 Workflow Agent"`
- Login/Register form using `/v1/auth/register` and `/v1/auth/login`
- Store `access_token` in `st.session_state`
- After login: show integration pills using `status_pill()` from `styles.py`
- Session dropdown from `/v1/sessions`
- “New session” button

Tabs:
- `Chat`
- `Plan Inspector`
- `History`

Chat tab:
- Render messages with `st.chat_message`
- On submit, call `POST /v1/invoke`
- Open SSE stream from `/v1/invoke/stream/{trace_id}` with `httpx`
- Parse `data:` lines in a loop
- Show inline status pills:
  - `📥 retrieving`
  - `🧠 planning`
  - `🔧 executing`
  - `✍️ composing`
  - `🛡️ checking`
- If `requires_approval=True`, show HITL UI

HITL UI:
- Use `st.expander` or bordered `st.container`
- Show draft summary + detail markdown in diff-like view
- Buttons: **Approve**, **Edit**, **Reject**
- Edit uses `st.text_area`
- On confirm, call `POST /v1/approvals/{token}` and reopen SSE stream

Plan Inspector:
- `st.json(state.plan.model_dump())`
- `st.metric("Confidence", f"{state.confidence:.0%}")`
- Table of `retrieved_plans` with `summary`, `tools_used`, `similarity`

History:
- Fetch `/v1/sessions`
- Render table with `started_at`, `last_activity`, `status badge`
- Clicking a row loads that session’s messages into Chat

Use only basic Streamlit primitives. No heavy component libraries.

---

## Do NOT Create

These files are owned by the Sonnet prompt:
- `app/config.py`, `app/logging.py`
- `app/security/*`, `app/data/*`
- `app/llm/client.py`, `app/llm/schemas.py`
- `app/prompts.py`
- `app/rag/qdrant_client.py`, `app/rag/embedder.py`, `app/rag/indexer.py`
- `app/mcp/servers/*`
- `app/agents/response_composer.py`
- `app/api/auth.py`, `app/api/health.py`, `app/api/sessions.py`, `app/api/feedback.py`, `app/api/integrations.py`
- `app/main.py`
- `streamlit_app/styles.py`
- `infra/*`, `scripts/*`

---

## Done When

- All 10 files exist with no `TODO` or `pass` placeholders.
- `app/core/state.py` exports all shared models cleanly.
- `compile_graph()` returns without error in a Python REPL.
- `make check` passes (`ruff`, `mypy`, `pytest`).
- Report the exact public interface — function signatures and return types — for each created file so the Sonnet prompt can import them safely.