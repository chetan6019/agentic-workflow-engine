"""LiteLLM-backed LangChain ChatOpenAI singleton factory."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import structlog
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.config import get_settings

log = structlog.get_logger(__name__)


@lru_cache(maxsize=8)
def get_llm(role: str) -> ChatOpenAI:
    """Return a cached ChatOpenAI client for the given LiteLLM role alias."""
    s = get_settings()
    log.debug("llm_client_created", role=role)
    return ChatOpenAI(
        model=role,
        openai_api_base=s.litellm_url,
        openai_api_key=s.litellm_virtual_key,
        timeout=30,
        max_retries=0,
    )


def get_structured_llm(role: str, schema: type) -> Runnable[Any, Any]:
    """Return a structured-output LLM bound to a Pydantic schema.

    Uses JSON mode (the model emits a JSON object parsed into the schema) rather
    than tool/function calling or strict json_schema. This avoids two failure
    modes seen with open models on Groq: corrupted tool names in function calling,
    and strict-schema rejection of open dict fields. Prompts must describe the
    expected JSON shape (see app/prompts.py).
    """
    return get_llm(role).with_structured_output(schema, method="json_mode")
