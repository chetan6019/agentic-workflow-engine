"""Application settings loaded from environment variables / .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All environment-variable-backed settings for the workflow engine.

    Fields are plain typed declarations — pydantic-settings reads them from the
    environment or ``.env`` (no manual ``os.getenv``/``load_dotenv``). Required
    fields have no default, so a missing one fails loudly at startup instead of
    silently becoming ``None`` (which also keeps ``mypy --strict`` happy).
    """

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # ── Required infrastructure ──────────────────────────────────────────────
    database_url: str
    redis_url: str
    qdrant_url: str
    litellm_url: str
    litellm_virtual_key: str
    fernet_key: str
    jwt_secret: str
    mcp_calendar_url: str
    mcp_gmail_url: str
    mcp_notion_url: str
    mcp_slack_url: str
    gmail_client_id: str
    gmail_client_secret: str

    # ── Optional / defaulted ─────────────────────────────────────────────────
    streamlit_origin: str = "http://localhost:8501"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    # Embedding client shape. "openai" speaks the OpenAI-compatible API to the
    # LiteLLM proxy (default); "hf" runs sentence-transformers in-process.
    embedding_provider: str
    hf_embedding_model: str = "BAAI/bge-base-en-v1.5"
    # IANA zone the calendar tools ask Google to return times in (and to interpret
    # event start/end). Defaults to IST; override with DEFAULT_TZ for other regions.
    default_tz: str = "Asia/Kolkata"
    # When true the retriever fuses dense + BM25 sparse hits with RRF; when false it
    # queries the dense named vector only. The collection schema always carries both
    # vectors, so flipping this needs no re-index.
    hybrid_search_enabled: bool = True
    # Max requests per identity (user_id, or client IP when unauthenticated) per
    # 60s window, enforced in the API middleware. Override with RATE_LIMIT_PER_MIN.
    rate_limit_per_min: int = 60
    # Hard ceiling on a single workflow run; a run exceeding it is marked
    # run_timeout instead of hanging forever. Override with RUN_TIMEOUT_SEC.
    run_timeout_sec: int = 120

    # ── New MCP servers: google / github / reddit / finnhub migration ────────
    # STALE (2026-06-22) context: the old mcp_{calendar,gmail,notion,slack}_url fields above
    # stay required and untouched per CLAUDE.md R13. These new fields are OPTIONAL with sane
    # defaults so the app keeps starting before .env is populated — a tool call fails only when
    # ITS specific credential is actually missing, not at import time.
    mcp_finnhub_url: str = "http://localhost:7005"
    mcp_reddit_url: str = "http://localhost:7006"
    mcp_github_url: str = "http://localhost:7007"
    mcp_google_url: str = "http://localhost:7008"
    # finnhub: API key only, read-only by design (no per-user OAuth token).
    finnhub_api_key: str | None = None
    # github: per-user OAuth token is stored via the integrations flow; these are the app creds.
    github_oauth_client_id: str | None = None
    github_oauth_client_secret: str | None = None
    # reddit: "script app" OAuth. client_id/secret drive app-only reads; username/password are
    # only needed for the HITL-gated write tools (post_comment / submit).
    reddit_client_id: str | None = None
    reddit_client_secret: str | None = None
    reddit_username: str | None = None
    reddit_password: str | None = None
    # google (Gmail + Calendar on one token). Falls back to gmail_client_id/secret above when
    # unset, so existing .env files keep working.
    google_oauth_client_id: str | None = None
    google_oauth_client_secret: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()  # type: ignore[call-arg]  # required fields come from env/.env
