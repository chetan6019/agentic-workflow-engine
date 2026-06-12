"""Application settings loaded from environment variables."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()  # Load environment variables from .env file if present

class Settings(BaseSettings):
    """All environment-variable-backed settings for the workflow engine."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    database_url: str = os.getenv("DATABASE_URL")
    redis_url: str = os.getenv("REDIS_URL")
    qdrant_url: str = os.getenv("QDRANT_URL")
    litellm_url: str = os.getenv("LITELLM_URL")
    litellm_virtual_key: str = os.getenv("LITELLM_VIRTUAL_KEY")
    fernet_key: str = os.getenv("FERNET_KEY")
    jwt_secret: str = os.getenv("JWT_SECRET")
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY")
    streamlit_origin: str = os.getenv("STREAMLIT_ORIGIN")
    mcp_calendar_url: str = os.getenv("MCP_CALENDAR_URL")
    mcp_gmail_url: str = os.getenv("MCP_GMAIL_URL")
    mcp_notion_url: str = os.getenv("MCP_NOTION_URL")
    mcp_slack_url: str = os.getenv("MCP_SLACK_URL")
    gmail_client_id: str = os.getenv("GMAIL_CLIENT_ID")
    gmail_client_secret: str = os.getenv("GMAIL_CLIENT_SECRET")
    # IANA zone the calendar tools ask Google to return times in (and to interpret
    # event start/end). Defaults to IST; override with DEFAULT_TZ for other regions.
    default_tz: str = os.getenv("DEFAULT_TZ", "Asia/Kolkata")
    # When true the retriever fuses dense + BM25 sparse hits with RRF; when false it
    # queries the dense named vector only. The collection schema always carries both
    # vectors, so flipping this needs no re-index.
    hybrid_search_enabled: bool = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() != "false"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
