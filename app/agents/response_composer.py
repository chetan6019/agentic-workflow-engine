"""Response composer agent — converts plan + tool results into a DraftResponse."""

from __future__ import annotations

from json import JSONDecodeError

import structlog
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from app.config import get_settings
from app.core.state import AgentState, DraftResponse, ToolSpec
from app.llm.client import get_structured_llm, run_metadata
from app.prompts import build_composer_messages, build_direct_answer_messages
from app.rag.embedder import embed_text
from app.rag.qdrant_client import DENSE_VECTOR, get_qdrant

log = structlog.get_logger(__name__)
_PREFS_COLLECTION = "preferences"
_TOP_PREFS = 3
_TOOL_DOC_COLLECTION = "tool_capability_docs"
_TOOL_DOC_K = 12
# Schema/JSON failures = LLM responded but the body was malformed → deterministic
# fallback is safe. Anything else (rate limits, timeouts, API errors, network)
# is a transport failure: re-raise so the orchestrator marks the run failed
# instead of fabricating a successful-looking draft from partial tool results.
_SCHEMA_ERRORS = (ValidationError, JSONDecodeError, OutputParserException)


async def _fetch_preferences(state: AgentState) -> list[str]:
    """Retrieve top-3 user preferences from Qdrant closest to the current request."""
    try:
        vec = await embed_text(state.user_request)
        hits = get_qdrant().query_points(
            collection_name=_PREFS_COLLECTION,
            query=vec,
            using=DENSE_VECTOR,
            query_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=state.user_id))]),
            limit=_TOP_PREFS,
        ).points
        return [h.payload.get("text", "") for h in hits if h.payload]
    except Exception as exc:
        log.warning("preferences_fetch_failed", error=str(exc))
        return []


def _fallback_draft(state: AgentState) -> DraftResponse:
    """Deterministic draft from tool results, used when the composer LLM fails."""
    ok = [r for r in state.tool_results if r.ok]
    failed = [r for r in state.tool_results if not r.ok]
    total = len(state.tool_results)
    summary = f"Completed {len(ok)}/{total} step(s)." if total else "No actions executed."
    lines = [f"**{summary}**", ""]
    for r in state.tool_results:
        mark = "✅" if r.ok else "❌"
        lines.append(f"{mark} `{r.step_id}`" + (f" — {r.error}" if r.error else ""))
    return DraftResponse(
        summary=summary,
        detail_markdown="\n".join(lines),
        actions_taken=[r.step_id for r in ok],
        actions_pending=[f"{r.step_id}: {r.error or 'failed'}" for r in failed],
        citations=[],
    )


def _inject_failures(draft: DraftResponse, state: AgentState) -> DraftResponse:
    """Ensure failed tool results are reflected honestly in actions_pending."""
    failed = [r for r in state.tool_results if not r.ok]
    if not failed:
        return draft
    pending = list(draft.actions_pending)
    for r in failed:
        entry = f"FAILED step {r.step_id}: {r.error or 'unknown error'} — please retry manually"
        if entry not in pending:
            pending.append(entry)
    return draft.model_copy(update={"actions_pending": pending})


async def _fetch_tool_catalog(user_request: str) -> list[ToolSpec]:
    """Vector-search Qdrant tool_capability_docs for the top tool specs.

    Mirrors the planner's helper so the direct-answer path knows the same
    capability surface as the planner would have considered.
    """
    try:
        vec = await embed_text(user_request)
        hits = get_qdrant().query_points(
            collection_name=_TOOL_DOC_COLLECTION, query=vec, using=DENSE_VECTOR, limit=_TOOL_DOC_K
        ).points
    except Exception as exc:
        log.warning("direct_answer_tool_catalog_failed", error=str(exc))
        return []
    specs: list[ToolSpec] = []
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        try:
            specs.append(ToolSpec.model_validate(payload))
        except Exception:
            continue
    return specs


async def _compose_direct_answer(state: AgentState, preferences: list[str],
                                 judge_feedback: str | None = None) -> DraftResponse:
    """Answer a meta/conversational request directly when no tools were involved."""
    tool_specs = await _fetch_tool_catalog(state.user_request)
    messages = build_direct_answer_messages(
        user_request=state.user_request,
        tool_specs=tool_specs,
        preferences=preferences,
        judge_feedback=judge_feedback,
    )
    llm = get_structured_llm("composer", DraftResponse, run_metadata(state))
    try:
        draft: DraftResponse = await llm.ainvoke(messages)
    except _SCHEMA_ERRORS as exc:
        log.warning("direct_answer_schema_fallback",
                    trace_id=state.trace_id, error=str(exc))
        draft = DraftResponse(
            summary="I couldn't generate a response.",
            detail_markdown=(
                "I'm not sure how to answer that. Try rephrasing as a task — "
                "for example, *'find unread emails from this week'* or "
                "*'list today's calendar events'*."
            ),
            actions_taken=[],
            actions_pending=[],
            citations=[],
        )
    log.info("direct_answer_composed", trace_id=state.trace_id)
    return draft


async def compose(state: AgentState, judge_feedback: str | None = None) -> DraftResponse:
    """Build a DraftResponse from the current AgentState.

    For meta/conversational queries (no plan steps, no tool results) the
    composer switches to a direct-answer prompt that uses the tool catalog as
    its grounding. Otherwise it summarises tool results. LLM transport errors
    propagate so the orchestrator can mark the run as failed; only
    schema-validation failures fall back to a deterministic draft.

    ``judge_feedback`` (set on a guard re-composition) is injected as a revision
    note instructing the model to ground every claim in the tool results.
    """
    structlog.contextvars.bind_contextvars(trace_id=state.trace_id, user_id=state.user_id)
    preferences = await _fetch_preferences(state)

    no_steps = state.plan is None or not state.plan.steps
    if no_steps and not state.tool_results:
        return await _compose_direct_answer(state, preferences, judge_feedback)

    messages = build_composer_messages(
        plan=state.plan,
        tool_results=state.tool_results,
        preferences=preferences,
        user_request=state.user_request,
        tz_name=get_settings().default_tz,
        judge_feedback=judge_feedback,
    )
    llm = get_structured_llm("composer", DraftResponse, run_metadata(state))
    try:
        draft: DraftResponse = await llm.ainvoke(messages)
    except _SCHEMA_ERRORS as exc:
        log.warning("composer_schema_fallback", trace_id=state.trace_id, error=str(exc))
        draft = _fallback_draft(state)
    draft = _inject_failures(draft, state)
    log.info("draft_composed", trace_id=state.trace_id,
             actions_pending=len(draft.actions_pending))
    return draft
