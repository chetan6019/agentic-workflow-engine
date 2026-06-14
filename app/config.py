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
    embedding_provider: Literal["openai", "hf"] = "openai"
    hf_embedding_model: str = "BAAI/bge-base-en-v1.5"
    # IANA zone the calendar tools ask Google to return times in (and to interpret
    # event start/end). Defaults to IST; override with DEFAULT_TZ for other regions.
    default_tz: str = "Asia/Kolkata"
    # When true the retriever fuses dense + BM25 sparse hits with RRF; when false it
    # queries the dense named vector only. The collection schema always carries both
    # vectors, so flipping this needs no re-index.
    hybrid_search_enabled: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()  # type: ignore[call-arg]  # required fields come from env/.env
