"""Agentic RAG retriever: route (decide+rewrite) → search → grade (with broadening fallback)."""

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
from app.llm.client import get_structured_llm, run_metadata
from app.prompts import (
    build_retriever_grader_messages,
    build_retriever_router_messages,
)
from app.rag.embedder import embed_sparse, embed_text
from app.rag.qdrant_client import DENSE_VECTOR, SPARSE_VECTOR, get_qdrant

log = structlog.get_logger(__name__)
_PLANS_COLLECTION = "plans"
_SEARCH_K = 10
_TOP_N = 5
_MIN_AFTER_GRADE = 3
_RELEVANCE_THRESHOLD = 0.7


class _RouteDecision(BaseModel):
    """Combined should-retrieve gate + rewritten search query (one LLM call)."""

    model_config = ConfigDict(extra="forbid")

    should_retrieve: bool = Field(description="True if past-plan retrieval would help.")
    query: str = Field(description="Concise keyword-rich search query.")


class _GradedHit(BaseModel):
    """Per-candidate relevance score returned by the grader."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str = Field(description="Candidate plan_id.")
    relevance: float = Field(ge=0.0, le=1.0, description="Relevance to user request.")


class _GraderOutput(BaseModel):
    """Wrapper holding a list of graded hits."""

    model_config = ConfigDict(extra="forbid")

    hits: list[_GradedHit] = Field(description="Per-candidate grades.")


async def _route_query(state: AgentState, metadata: dict[str, str]) -> _RouteDecision:
    """One structured call: decide IF retrieval helps AND rewrite the search query.

    Replaces the old two-call decide → rewrite preamble, removing one LLM
    round-trip from every request's critical path. On any LLM failure, fall
    back to retrieving with the raw request — retrieval is best-effort and
    must never be run-fatal.
    """
    msgs = build_retriever_router_messages(user_request=state.user_request)
    llm = get_structured_llm("retriever-grader", _RouteDecision, metadata)
    try:
        decision: _RouteDecision = await llm.ainvoke(msgs)
    except Exception as exc:
        log.warning("retrieve_route_fallback", error=str(exc))
        return _RouteDecision(should_retrieve=True, query=state.user_request)
    if not decision.query.strip():
        decision.query = state.user_request
    return decision


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
    # TODO(REVIEW.md R26): the fused and dense passes below run sequentially
    # because QdrantClient is sync; switch to AsyncQdrantClient and gather them.
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
    # Emit per-candidate relevance + rank so retrieval quality is observable
    # online (and joinable with feedback/eval offline — REVIEW.md R29). Ranked
    # by the grader's own score, highest first; threshold drives the keep set.
    ranked = sorted(graded.hits, key=lambda h: h.relevance, reverse=True)
    log.info("retrieve_graded",
             scores=[{"plan_id": h.plan_id, "relevance": round(h.relevance, 3),
                      "rank": i + 1, "kept": h.relevance >= _RELEVANCE_THRESHOLD}
                     for i, h in enumerate(ranked)])
    keep = {h.plan_id for h in graded.hits if h.relevance >= _RELEVANCE_THRESHOLD}
    return [c for c in candidates if c.plan_id in keep]


async def retrieve(state: AgentState) -> list[RetrievedPlan]:
    """Run the agentic retrieval loop and return up to 5 graded RetrievedPlans."""
    meta = run_metadata(state)
    route = await _route_query(state, meta)
    if not route.should_retrieve:
        log.info("retrieve_skipped", trace_id=state.trace_id)
        return []
    candidates = await _search(route.query, {"user_id": state.user_id, "success": True})
    graded = await _grade(candidates, state.user_request, meta)
    log.info("retrieve_user_scoped", candidates=len(candidates), graded=len(graded))
    if len(graded) < _MIN_AFTER_GRADE:
        broad = await _search(route.query, {"success": True})
        graded = await _grade(broad, state.user_request, meta)
        log.info("retrieve_broadened", candidates=len(broad), graded=len(graded))
    # Order by the fused score so RRF's blend wins; in dense-only mode fused_score
    # equals the cosine, so this stays correct either way.
    graded.sort(key=lambda p: p.fused_score, reverse=True)
    return graded[:_TOP_N]
