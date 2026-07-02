"""LiteLLM-backed LangChain ChatOpenAI factory with per-run trace metadata."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.core.metrics import llm_call_cost_usd_total, llm_call_tokens_total
from app.core.state import AgentState
from app.prompts import PROMPT_VERSION, fingerprint

log = structlog.get_logger(__name__)

# USD per 1M tokens. Hardcoded because Groq's gpt-oss-* isn't in LiteLLM's
# default cost table at time of writing, so response_cost comes back as 0.
# Update when prices move or new models are added; keep keys identical to the
# `model` field LiteLLM emits in response metadata.
_PRICE_PER_M_TOKENS: dict[str, dict[str, float]] = {
    "groq/openai/gpt-oss-120b": {"prompt": 0.15, "completion": 0.60},
    "groq/openai/gpt-oss-20b":  {"prompt": 0.05, "completion": 0.20},
    "gemini/gemini-2.5-flash":  {"prompt": 0.075, "completion": 0.30},
    "openai/gpt-4o-mini":       {"prompt": 0.15, "completion": 0.60},
}


def _compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return USD cost for a single LLM call, or 0.0 if the model isn't priced.

    Returns 0 (instead of raising) when the model is unknown so an unpriced model
    silently shows 0 cost in the dashboard rather than breaking the whole metric.
    Check llm_client_unknown_model_price logs to know when to add a new row above.
    """
    prices = _PRICE_PER_M_TOKENS.get(model)
    if not prices:
        log.debug("llm_client_unknown_model_price", model=model)
        return 0.0
    return (prompt_tokens * prices["prompt"] + completion_tokens * prices["completion"]) / 1_000_000


class _LLMMeteringCallback(BaseCallbackHandler):
    """Observe per-role token + cost counters when an LLM call finishes.

    Subclasses BaseCallbackHandler (sync interface). LangChain dispatches `on_llm_end`
    even for async calls, so one handler covers both paths.

    Instances are bound to a single `role` because that label has to come from
    outside the LLM response. One callback per role per call site.
    """

    # Tell LangChain the handler is safe to run inline (it's just a counter inc,
    # no I/O) so it doesn't get pushed onto a thread pool.
    raise_error: bool = False
    run_inline: bool = True

    def __init__(self, role: str) -> None:
        self.role = role

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # `llm_output` shape varies slightly by provider; LangChain-OpenAI puts
        # token_usage there. Defensive .get() chain keeps this from raising even
        # if a future provider returns a different envelope.
        info = response.llm_output or {}
        usage = info.get("token_usage") or {}
        model = info.get("model_name") or "unknown"

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)

        if prompt_tokens:
            llm_call_tokens_total.labels(
                role=self.role, model=model, kind="prompt"
            ).inc(prompt_tokens)
        if completion_tokens:
            llm_call_tokens_total.labels(
                role=self.role, model=model, kind="completion"
            ).inc(completion_tokens)

        cost = _compute_cost_usd(model, prompt_tokens, completion_tokens)
        if cost > 0:
            llm_call_cost_usd_total.labels(role=self.role, model=model).inc(cost)

PROMPT_CONFIGS: dict[str, dict[str, Any]] = {
    "planner":  {"temperature": 0.1, "max_tokens": 1200},
    "guard":    {"temperature": 0.0, "max_tokens": 200},
    "composer": {"temperature": 0.4, "max_tokens": 1500},
    "router":   {"temperature": 0.0, "max_tokens": 80},
    "grader":   {"temperature": 0.0, "max_tokens": 200},
    "direct":   {"temperature": 0.5, "max_tokens": 1200},
}

@lru_cache(maxsize=8)
def _base_llm(role: str) -> ChatOpenAI:
    """Return the cached ChatOpenAI client for the given LiteLLM role alias.

    Per-role temperature and max_tokens come from PROMPT_CONFIGS (declared above)
    so each prompt's sampling profile is colocated with the other prompt-level
    knobs. Roles not in PROMPT_CONFIGS get LangChain's defaults — no error, just
    a debug log so it's visible in traces if a new role wasn't configured.
    """
    s = get_settings()
    cfg = PROMPT_CONFIGS.get(role, {})
    if not cfg:
        log.debug("llm_client_no_prompt_config", role=role)
    log.debug("llm_client_created", role=role, **cfg)
    return ChatOpenAI(
        model=role,
        openai_api_base=s.litellm_url,
        openai_api_key=s.litellm_virtual_key,
        timeout=30,
        max_retries=0,
        **cfg,
    )


def run_metadata(state: AgentState) -> dict[str, str]:
    """Per-run metadata for get_llm/get_structured_llm, so LLM traces join app traces."""
    return {"trace_id": state.trace_id, "session_id": state.session_id,
            "user_id": state.user_id}


def call_metadata(state: AgentState, messages: list[BaseMessage]) -> dict[str, str]:
    """Per-CALL metadata = run_metadata + this call's prompt fingerprint.

    Call this from sites that have the rendered messages in hand (build_planner_messages,
    build_guard_judge_messages, etc.) and pass the result to get_llm. The fingerprint
    groups identical prompt renderings together in Langfuse regardless of run/session;
    PROMPT_VERSION is attached separately in _apply_metadata so it's always present.
    Use run_metadata instead when messages aren't available at the call site (rare).
    """
    return {**run_metadata(state), "prompt_fingerprint": fingerprint(messages)}


def _apply_metadata(llm: ChatOpenAI, role: str, metadata: dict[str, str]) -> ChatOpenAI:
    """Copy the cached client with a LiteLLM extra_body carrying the run's metadata.

    The metadata keys are LiteLLM's Langfuse grouping fields (trace_id,
    session_id, trace_user_id, generation_name), so every LLM hop of one
    workflow run lands in a single Langfuse trace with generations named by
    role. For cost attribution: body-level `user` drives LiteLLM's per-end-user
    spend, and `metadata.tags = ["role:<role>"]` drives its per-tag spend
    (/spend/tags), so planner vs. judge cost separates without per-role keys.

    `prompt_version` is injected unconditionally so every trace can be attributed
    to an exact app/prompts.py revision; `prompt_fingerprint` is forwarded from
    callers that used call_metadata() and is omitted when only run_metadata()
    was used (no message context available).

    model_copy is shallow: the underlying HTTP client and its connection pool
    stay shared with the cached instance.
    """
    md: dict[str, Any] = {
        "generation_name": role,
        "tags": [f"role:{role}", f"prompt_version:{PROMPT_VERSION}"],
        "prompt_version": PROMPT_VERSION,
        **metadata,
    }
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

    The metering callback is attached AFTER with_structured_output so it observes
    the raw LLM call (where token_usage is populated) — not the post-parse step.
    """
    base = get_llm(role, metadata)
    structured = base.with_structured_output(schema, method="json_mode")
    return structured.with_config(callbacks=[_LLMMeteringCallback(role=role)])
