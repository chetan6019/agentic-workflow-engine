# Claude Code Prompt — Sonnet 4.6
# Role: Implement all supporting, utility, and leaf-level files.
# Run this AFTER the Opus 4.7 prompt has completed.
# `app/core/state.py` and all orchestration files already exist. Import from them; do not recreate them.

---

## Read First

Before writing a single line of code:
1. Read `project_prompt.md` — full architecture reference.
2. Read `CLAUDE.md` — project rules, non-negotiables, and working style.
3. Read `app/core/state.py` — all shared Pydantic models are defined there. Import from it everywhere you need `AgentState`, `ExecutionPlan`, `DraftResponse`, `ToolResult`, `RetrievedPlan`, `ToolSpec`.

Confirm you have read all three before proceeding.

---

## Framework Mandate

The latest project rules are non-negotiable:

- **LangChain** is the primary framework for all agent chains, prompts, embeddings, vector store integration, structured outputs, and output parsing.
- **LangGraph** is the required framework for all stateful multi-step orchestration.
- **Qdrant** is the required vector database for plan retrieval, preference retrieval, and tool capability retrieval.
- **langchain-mcp-adapters** should be used where it meaningfully improves MCP-to-LangChain/LangGraph interoperability.
- **LiteLLM** is the only allowed gateway for chat and embedding model access.

Implement every supporting file in a way that reinforces those decisions rather than working around them.

---

## Constraints (do not violate)

- Every file must be under 150 lines. Split into sub modules within the same file if needed.
- No imports of `openai`, `anthropic`, or any provider SDK. Use `app/llm/client.py:get_llm(role)` only.
- Use **LangChain** for prompt/message construction, structured outputs, embeddings, and vector-store-facing integrations.
- Do not replace **Qdrant** with any other vector store.
- Use `langchain-mcp-adapters` only where it helps interoperability; do not replace the custom MCP client/server architecture defined by the project.
- Pydantic v2 everywhere. External-facing schemas use `model_config = ConfigDict(extra="forbid")`.
- Async-first. Use `asyncpg` and `httpx.AsyncClient`. Never `requests`. No blocking I/O.
- Singletons via `@lru_cache` for: `AsyncEngine`, Redis pool, Qdrant client, Fernet cipher, `ChatOpenAI` per role.
- Every function must have a 1–2 line meaningful docstring about its functionality.
- No Claude models. Default is `gpt-4o-mini` via LiteLLM alias. Escalation to `gpt-4o` or approved Groq alias only where already defined by the architecture.
- Keep modules small, flat, interview-friendly, and easy to explain.

---

## Files to Create

Create each file completely before moving to the next. Run `make check` after every 5 files.

### Config & Logging

1. `app/config.py`
- Pydantic `BaseSettings` class named `Settings`.
- Fields:
  - `DATABASE_URL`
  - `REDIS_URL`
  - `QDRANT_URL`
  - `LITELLM_URL`
  - `LITELLM_VIRTUAL_KEY`
  - `FERNET_KEY`
  - `JWT_SECRET`
  - `LANGFUSE_PUBLIC_KEY`
  - `LANGFUSE_SECRET_KEY`
  - `STREAMLIT_ORIGIN`
  - MCP server URLs for calendar, gmail, notion, slack
- Single `@lru_cache` getter `get_settings() -> Settings`.

2. `app/logging.py`
- Configure `structlog` to output JSON to stdout.
- Processors:
  - `TimeStamper(fmt="iso")`
  - `add_log_level`
  - `merge_contextvars`
  - `format_exc_info`
- One `configure_logging()` function called once at startup.

### Security

3. `app/security/passwords.py`
- `hash_password(plain: str) -> str` — bcrypt, cost 12.
- `verify_password(plain: str, hashed: str) -> bool`.

4. `app/security/jwt_tokens.py`
- `create_access_token(user_id: str) -> str` — HS256, 24h expiry, `python-jose`.
- `decode_access_token(token: str) -> str` — returns `user_id`; raises `HTTPException(401)` if expired or invalid.

5. `app/security/crypto.py`
- `get_fernet() -> Fernet` — `@lru_cache` singleton using `FERNET_KEY`.
- `encrypt(text: str) -> str`
- `decrypt(token_enc: str) -> str`

### Data Layer

6. `app/data/models.py`
- SQLAlchemy 2.0 ORM mapped classes for:
  - `users`
  - `sessions`
  - `messages`
  - `plans`
  - `tool_calls`
  - `approvals`
  - `integration_tokens`
  - `feedback`
  - `audit_log`
- Use `mapped_column`, `Mapped`, `relationship`.

7. `app/data/db.py`
- `get_engine() -> AsyncEngine` — `@lru_cache`, `asyncpg` driver, reads `DATABASE_URL`.
- `get_async_session()` — async context manager returning `AsyncSession`.
- `init_db()` — create all tables for startup use.

8. `app/data/redis_client.py`
- `get_redis() -> Redis` — `@lru_cache` singleton, reads `REDIS_URL`.
- `set_idempotency_key(key: str, ttl: int = 600)`
- `check_idempotency_key(key: str) -> bool`

9. `app/data/repositories.py`
- Flat async functions, grouped by entity.
- Users:
  - `create_user`
  - `get_user_by_username`
  - `get_user_by_id`
- Sessions:
  - `create_session`
  - `get_sessions_by_user`
  - `get_session`
- Plans:
  - `save_plan`
  - `get_plan_by_trace_id`
- Approvals:
  - `create_approval`
  - `get_approval_by_token`
  - `update_approval_status`
- Feedback:
  - `save_feedback`
- Integration tokens:
  - `save_token`
  - `get_token`
- Audit log:
  - `log_action`
- Do not return ORM instances; return plain dicts or lightweight dataclasses.

### LLM Client

10. `app/llm/client.py`
- `get_llm(role: str) -> ChatOpenAI`
- `@lru_cache(maxsize=8)`
- Points at `LITELLM_URL`
- Uses `LITELLM_VIRTUAL_KEY`
- `max_retries=0`
- This is the only file in the repo allowed to construct an LLM client.

11. `app/llm/schemas.py`
- `redact_pii(text: str) -> str`
- Regex-based masking for:
  - emails
  - phone numbers
  - SSNs in `XXX-XX-XXXX`
  - credit card patterns
- Replace matches with `[REDACTED]`
- Only used on copies sent to logs/traces, never on live LLM input.

### Prompts

12. `app/prompts.py`
- Single file.
- Pure builder functions only: no imports of LLM clients, DB, or HTTP.
- All prompts use Anthropic-style XML tags:
  - `<instructions>`
  - `<examples>`
  - `<request>`
  - `<user_style>`
  - `<tools>`

Required builders:
- `build_planner_messages(user_request, examples, tool_specs) -> list[dict]`
- `build_retriever_rewriter_messages(user_request) -> list[dict]`
- `build_retriever_grader_messages(query, candidates) -> list[dict]`
- `build_guard_judge_messages(draft, user_request) -> list[dict]`
- `build_composer_messages(plan, tool_results, preferences, user_request) -> list[dict]`

Rules:
- Full system prompt constants at module level.
- Output should be LangChain-friendly message lists.
- Keep prompts centralized and pure, matching project rules.

### RAG Utilities

13. `app/rag/qdrant_client.py`
- `get_qdrant() -> QdrantClient` — `@lru_cache` singleton, reads `QDRANT_URL`.
- `ensure_collections()` — idempotently creates:
  - `plans`
  - `preferences`
  - `tool_capability_docs`
- Use cosine distance, 1536 dimensions, HNSW `m=16`.

14. `app/rag/embedder.py`
- `embed_text(text: str) -> list[float]`
- Calls `get_llm("embed")` through the gateway.
- Caches results in Redis with key:
  - `emb:{sha256(normalized_text)}`
- TTL: 86400 seconds.
- Returns the embedding vector.

15. `app/rag/indexer.py`
- `index_plan(state: AgentState)`
- Embeds `(user_request + plan_summary)` and upserts to Qdrant `plans`.
- `index_preference(user_id: str, kind: str, text: str)`
- Upserts to `preferences`.
- Called as a background task after successful workflow runs.

### MCP Servers

16. `app/mcp/servers/_shared.py`
- `resolve_user_token(user_id: str, provider: str) -> str`
- Decrypts token from `integration_tokens` via `crypto.decrypt`
- `http_client(timeout: int = 15)`
- Returns a context-managed `httpx.AsyncClient`

17. `app/mcp/servers/calendar_server.py`
- FastMCP app named `"calendar"`
- Port `:7001`
- Tools with Pydantic v2 input models:
  - `create_event`
  - `update_event`
  - `delete_event`
  - `find_free_slot`
  - `list_upcoming`
- Each tool calls Google Calendar via `httpx`
- No LLM calls

18. `app/mcp/servers/gmail_server.py`
- Port `:7002`
- Tools:
  - `send_email`
  - `search_email`
  - `create_draft`
  - `reply_thread`

19. `app/mcp/servers/notion_server.py`
- Port `:7003`
- Tools:
  - `create_page`
  - `append_block`
  - `search_pages`
  - `update_page`

20. `app/mcp/servers/slack_server.py`
- Port `:7004`
- Tools:
  - `send_message`
  - `schedule_message`
  - `search_messages`
  - `list_channels`

Rules for all MCP servers:
- Deterministic adapters only.
- No LLM calls.
- Small focused files.
- `langchain-mcp-adapters` compatibility is welcome where helpful for schema/tool interoperability, but do not redesign the MCP architecture around it.

### Agent

21. `app/agents/response_composer.py`
- `async def compose(state: AgentState) -> DraftResponse`
- Fetch top-3 preferences from Qdrant `preferences`
- Call `build_composer_messages(...)`
- Call `get_llm("composer").with_structured_output(DraftResponse).ainvoke(messages)`
- If any `tool_results[i].ok is False`, reflect that in `actions_pending`
- Never fabricate success

### API Routers

22. `app/api/auth.py`
- `POST /v1/auth/register`
  - hash password
  - insert user
  - return `{user_id, access_token}`
- `POST /v1/auth/login`
  - verify bcrypt
  - return `{access_token}`
- `GET /v1/auth/me`
  - return current user from JWT

23. `app/api/health.py`
- `GET /healthz` → `{"status": "ok"}`
- `GET /readyz`
  - check Postgres
  - check Redis
  - check Qdrant
  - return `{"status": "ok"}` or `503`

24. `app/api/sessions.py`
- `GET /v1/sessions`
  - list sessions for authenticated user
- `GET /v1/sessions/{id}`
  - return session + its messages

25. `app/api/feedback.py`
- `POST /v1/feedback/{trace_id}`
- Accepts `{score: int, comment?: str}`
- Writes to `feedback`

26. `app/api/integrations.py`
- `POST /v1/integrations/{provider}/token`
- Accepts `{token: str}`
- Fernet-encrypts and upserts to `integration_tokens`

### App Factory

27. `app/main.py`
- FastAPI app factory:
  - `create_app() -> FastAPI`
- Register all routers from `app/api/`
- Add middleware in this exact order:
  1. JWT auth
  2. per-user rate limit
  3. CORS
- Call `configure_logging()` and `init_db()` in startup
- `app = create_app()` at module level for Uvicorn

### Frontend Styles

28. `streamlit_app/styles.py`
- `inject_styles()`
  - one `st.markdown` CSS block
  - rounded corners
  - softer shadows
  - tighter padding
  - chat bubbles:
    - user: right-aligned violet `#7C3AED`
    - assistant: left-aligned slate
- `status_pill(label: str, color: str) -> str`
  - returns HTML span styled as a colored pill badge

### Infrastructure

29. `infra/litellm_config.yaml`
- Pin model aliases:
  - `planner-default` → `openai/gpt-4o-mini-2024-07-18`
  - `planner-escalation` → `openai/gpt-4o-2024-08-06` or approved Groq alias
  - `composer` → `openai/gpt-4o-mini-2024-07-18`
  - `retriever-grader` → `openai/gpt-4o-mini-2024-07-18`
  - `retriever-rewriter` → `openai/gpt-4o-mini-2024-07-18`
  - `guard-judge` → `openai/gpt-4o-mini-2024-07-18`
  - `embed` → `openai/text-embedding-3-small`
- Redis cache with per-role TTLs:
  - planner 10 min
  - composer 5 min
  - embeddings 24h
- Langfuse success + failure callbacks
- Per-model timeouts:
  - embed 10s
  - chat 30s
  - heavy 60s
- `num_retries: 2`
- `disable_unhealthy_models: true`

30. `infra/docker-compose.yml`
- Services:
  - `api`
  - `streamlit`
  - `litellm-proxy`
  - `mcp-calendar`
  - `mcp-gmail`
  - `mcp-notion`
  - `mcp-slack`
  - `qdrant`
  - `postgres`
  - `redis`
- Port mappings must match the project spec
- Add `healthcheck` on `api` using `/readyz`
- All secrets via environment variables
- Include `env_file: .env`

### Scripts

31. `scripts/seed_demo_data.py`
- Creates one demo user:
  - username `demo`
  - password `demo123`
- Seeds 25 example past plans into Qdrant `plans`
- Seeds Slack, Calendar, Gmail, Notion tokens from env vars

32. `scripts/index_tool_docs.py`
- Reads tool docstrings from all four MCP server files using `importlib`
- Embeds each tool description via `app/rag/embedder.embed_text`
- Upserts into Qdrant `tool_capability_docs`

---

## Do NOT Create

These files are owned by the Opus prompt. Do not create or modify them:
- `app/core/state.py`
- `app/orchestration/graph.py`
- `app/orchestration/nodes/orchestrator.py`
- `app/orchestration/nodes/planner.py`
- `app/orchestration/nodes/guard.py`
- `app/rag/retriever.py`
- `app/mcp/client.py`
- `app/api/invoke.py`
- `app/api/approvals.py`
- `streamlit_app/app.py`

---

## Done When

- All 32 files exist with no `TODO` or `pass` placeholders.
- `make check` passes (`ruff`, `mypy`, `pytest`).
- `make up && curl localhost:8000/readyz` returns `{"status": "ok"}`.
- Report a one-line summary of each file created.