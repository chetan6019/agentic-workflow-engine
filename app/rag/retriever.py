"""Agentic RAG retriever: decide → rewrite → search → grade (with broadening fallback)."""

from __future__ import annotations

import time
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
)

from app.config import get_settings
from app.core.state import AgentState, RetrievedPlan
from app.llm.client import get_llm, get_structured_llm, run_metadata
from app.prompts import (
    build_retriever_grader_messages,
    build_retriever_rewriter_messages,
)
from app.rag.embedder import embed_sparse, embed_text
from app.rag.qdrant_client import DENSE_VECTOR, SPARSE_VECTOR, get_qdrant

log = structlog.get_logger(__name__)
_PLANS_COLLECTION = "plans"
_SEARCH_K = 10
_TOP_N = 5
_MIN_AFTER_GRADE = 3
_RELEVANCE_THRESHOLD = 0.7


class _RetrieveDecision(BaseModel):
    """Structured yes/no decision from the retriever-grader."""

    model_config = ConfigDict(extra="forbid")

    should_retrieve: bool = Field(description="True if retrieval would help.")
    reason: str = Field(description="One-line rationale.")


class _GradedHit(BaseModel):
    """Per-candidate relevance score returned by the grader."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(description="Candidate plan_id.")
    relevance: float = Field(ge=0.0, le=1.0, description="Relevance to user request.")


class _GraderOutput(BaseModel):
    """Wrapper holding a list of graded hits."""

    model_config = ConfigDict(extra="forbid")

    hits: list[_GradedHit] = Field(description="Per-candidate grades.")


async def _should_retrieve(state: AgentState) -> bool:
    """Cheap LLM gate to decide whether retrieval is worth running."""
    msgs = build_retriever_grader_messages(query=state.user_request, candidates=[])
    llm = get_structured_llm("retriever-grader", _RetrieveDecision, run_metadata(state))
    try:
        decision: _RetrieveDecision = await llm.ainvoke(msgs)
    except Exception:
        return True
    return decision.should_retrieve


async def _rewrite_query(user_request: str, metadata: dict[str, str]) -> str:
    """Rewrite the user request into a denser search query."""
    msgs = build_retriever_rewriter_messages(user_request=user_request)
    llm = get_llm("retriever-rewriter", metadata)
    out = await llm.ainvoke(msgs)
    return (getattr(out, "content", None) or str(out)).strip() or user_request


def _build_filter(filters: dict[str, Any] | None) -> Filter | None:
    """Turn a flat field→value dict into a Qdrant must-match Filter."""
    if not filters:
        return None
    return Filter(must=[
        FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()
    ])


def _to_plan(hit: Any, similarity: float, fused_score: float) -> RetrievedPlan | None:
    """Build a RetrievedPlan from a Qdrant hit, or None if its payload is malformed."""
    payload = getattr(hit, "payload", None) or {}
    try:
        return RetrievedPlan(
            plan_id=str(getattr(hit, "id", payload.get("plan_id", ""))),
            request_text=payload.get("request_text", ""),
            plan_json=payload.get("plan_json", {}),
            summary=payload.get("summary", ""),
            # similarity must stay in cosine space — it feeds the guardrails formula.
            similarity=max(0.0, min(1.0, similarity)),
            fused_score=fused_score,
        )
    except Exception:
        return None


async def _dense_only_search(
    vec: list[float], qfilter: Filter | None, k: int
) -> list[RetrievedPlan]:
    """Plain cosine search against the dense named vector (hybrid disabled or fallback)."""
    t0 = time.monotonic()
    hits = get_qdrant().query_points(
        collection_name=_PLANS_COLLECTION,
        query=vec,
        using=DENSE_VECTOR,
        query_filter=qfilter,
        limit=k,
    ).points
    plans = [_to_plan(h, float(getattr(h, "score", 0.0)), float(getattr(h, "score", 0.0)))
             for h in hits]
    plans = [p for p in plans if p is not None]
    log.info("search_timing", mode="dense", total_ms=int((time.monotonic() - t0) * 1000),
             hits=len(plans))
    return plans


async def _search(query: str, filters: dict[str, Any] | None, k: int = _SEARCH_K) -> list[RetrievedPlan]:
    """Search the plans collection: RRF-fused dense+sparse when hybrid is on, else cosine.

    Ordering comes from the fused score, but ``similarity`` always carries the dense
    cosine (sourced from a parallel dense pass) so the guardrails confidence math stays
    in cosine space.
    """
    vec = await embed_text(query)
    qfilter = _build_filter(filters)
    if not get_settings().hybrid_search_enabled:
        return await _dense_only_search(vec, qfilter, k)

    t0 = time.monotonic()
    sparse = await embed_sparse(query)
    t1 = time.monotonic()
    client = get_qdrant()
    fused = client.query_points(
        collection_name=_PLANS_COLLECTION,
        prefetch=[
            Prefetch(query=vec, using=DENSE_VECTOR, filter=qfilter, limit=k),
            Prefetch(query=sparse, using=SPARSE_VECTOR, filter=qfilter, limit=k),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
    ).points
    t2 = time.monotonic()
    # RRF drops per-vector scores, so run one dense pass to recover cosine per hit.
    dense = client.query_points(
        collection_name=_PLANS_COLLECTION,
        query=vec,
        using=DENSE_VECTOR,
        query_filter=qfilter,
        limit=k,
    ).points
    t3 = time.monotonic()
    cosine = {str(getattr(h, "id", "")): float(getattr(h, "score", 0.0)) for h in dense}
    plans: list[RetrievedPlan] = []
    for hit in fused:
        hid = str(getattr(hit, "id", ""))
        plan = _to_plan(hit, cosine.get(hid, 0.0), float(getattr(hit, "score", 0.0)))
        if plan is not None:
            plans.append(plan)
    log.info("search_timing", mode="hybrid",
             sparse_ms=int((t1 - t0) * 1000), fused_ms=int((t2 - t1) * 1000),
             dense_ms=int((t3 - t2) * 1000), total_ms=int((t3 - t0) * 1000),
             hits=len(plans))
    return plans


async def _grade(candidates: list[RetrievedPlan], user_request: str,
                 metadata: dict[str, str]) -> list[RetrievedPlan]:
    """Ask the grader LLM to score candidates, drop ones below the threshold."""
    if not candidates:
        return []
    msgs = build_retriever_grader_messages(query=user_request, candidates=candidates)
    llm = get_structured_llm("retriever-grader", _GraderOutput, metadata)
    try:
        graded: _GraderOutput = await llm.ainvoke(msgs)
    except Exception:
        return candidates
    keep = {h.plan_id for h in graded.hits if h.relevance >= _RELEVANCE_THRESHOLD}
    return [c for c in candidates if c.plan_id in keep]


async def retrieve(state: AgentState) -> list[RetrievedPlan]:
    """Run the agentic retrieval loop and return up to 5 graded RetrievedPlans."""
    if not await _should_retrieve(state):
        log.info("retrieve_skipped", trace_id=state.trace_id)
        return []
    meta = run_metadata(state)
    query = await _rewrite_query(state.user_request, meta)
    candidates = await _search(query, {"user_id": state.user_id, "success": True})
    graded = await _grade(candidates, state.user_request, meta)
    log.info("retrieve_user_scoped", candidates=len(candidates), graded=len(graded))
    if len(graded) < _MIN_AFTER_GRADE:
        broad = await _search(query, {"success": True})
        graded = await _grade(broad, state.user_request, meta)
        log.info("retrieve_broadened", candidates=len(broad), graded=len(graded))
    # Order by the fused score so RRF's blend wins; in dense-only mode fused_score
    # equals the cosine, so this stays correct either way.
    graded.sort(key=lambda p: p.fused_score, reverse=True)
    return graded[:_TOP_N]
