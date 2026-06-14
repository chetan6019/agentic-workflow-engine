"""LiteLLM-backed LangChain ChatOpenAI factory with per-run trace metadata."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import structlog
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.core.state import AgentState

log = structlog.get_logger(__name__)


@lru_cache(maxsize=8)
def _base_llm(role: str) -> ChatOpenAI:
    """Return the cached ChatOpenAI client for the given LiteLLM role alias."""
    s = get_settings()
    log.debug("llm_client_created", role=role)
    return ChatOpenAI(
        model=role,
        openai_api_base=s.litellm_url,
        openai_api_key=s.litellm_virtual_key,
        timeout=30,
        max_retries=0,
    )


def run_metadata(state: AgentState) -> dict[str, str]:
    """Per-run metadata for get_llm/get_structured_llm, so LLM traces join app traces."""
    return {"trace_id": state.trace_id, "session_id": state.session_id,
            "user_id": state.user_id}


def _apply_metadata(llm: ChatOpenAI, role: str, metadata: dict[str, str]) -> ChatOpenAI:
    """Copy the cached client with a LiteLLM extra_body carrying the run's metadata.

    The metadata keys are LiteLLM's Langfuse grouping fields (trace_id,
    session_id, trace_user_id, generation_name), so every LLM hop of one
    workflow run lands in a single Langfuse trace with generations named by
    role. For cost attribution: body-level `user` drives LiteLLM's per-end-user
    spend, and `metadata.tags = ["role:<role>"]` drives its per-tag spend
    (/spend/tags), so planner vs. judge cost separates without per-role keys.
    model_copy is shallow: the underlying HTTP client and its connection pool
    stay shared with the cached instance.
    """
    md: dict[str, Any] = {"generation_name": role, "tags": [f"role:{role}"], **metadata}
    user_id = md.pop("user_id", None)
    if user_id:
        md["trace_user_id"] = user_id
    body: dict[str, Any] = {"metadata": md}
    if user_id:
        body["user"] = user_id
    return llm.model_copy(update={"extra_body": body})


def get_llm(role: str, metadata: dict[str, str] | None = None) -> ChatOpenAI:
    """Return a ChatOpenAI client for the role, with optional run metadata attached."""
    llm = _base_llm(role)
    if metadata:
        llm = _apply_metadata(llm, role, metadata)
    return llm


def get_structured_llm(
    role: str, schema: type, metadata: dict[str, str] | None = None
) -> Runnable[Any, Any]:
    """Return a structured-output LLM bound to a Pydantic schema.

    Uses JSON mode (the model emits a JSON object parsed into the schema) rather
    than tool/function calling or strict json_schema. This avoids two failure
    modes seen with open models on Groq: corrupted tool names in function calling,
    and strict-schema rejection of open dict fields. Prompts must describe the
    expected JSON shape (see app/prompts.py).
    """
    return get_llm(role, metadata).with_structured_output(schema, method="json_mode")
