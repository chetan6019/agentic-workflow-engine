"""Agentic RAG retriever: route (decide+rewrite) → search → grade (with broadening fallback)."""

from __future__ import annotations

import asyncio
import hashlib
import json
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
from app.data.redis_client import get_redis
from app.llm.client import get_structured_llm, run_metadata
from app.prompts import (
    build_retriever_grader_messages,
    build_retriever_router_messages,
)
from app.rag.embedder import embed_sparse, embed_text, rerank_scores
from app.rag.qdrant_client import DENSE_VECTOR, SPARSE_VECTOR, get_qdrant

log = structlog.get_logger(__name__)
_PLANS_COLLECTION = "plans"
_SEARCH_K = 10
_TOP_N = 3
_MIN_AFTER_GRADE = 3
_RELEVANCE_THRESHOLD = 0.75
# After a cross-encoder rerank, keep this many candidates for the (pricier) LLM grader.
# Must stay >= _MIN_AFTER_GRADE so the grade step still has enough to clear the threshold.
_RERANK_TOP_N = 5


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


def _estimate_tokens(plans: list[RetrievedPlan]) -> int:
    """Rough token estimate (chars // 4) of a candidate set's payload text.

    Good enough for observability — this feeds a structlog counter comparing how
    much text retrieval FETCHED vs what survived rerank+grade, not billing (exact
    per-call tokens stay LiteLLM's job). chars/4 is the standard English-text
    approximation and needs no tokenizer dependency.
    """
    chars = 0
    for p in plans:
        chars += len(p.request_text) + len(p.summary) + len(json.dumps(p.plan_json))
    return chars // 4


def _router_cache_key(user_request: str) -> str:
    """Redis key for a router decision, from the normalized request text.

    Normalizing (lowercase + collapsed whitespace) makes "Show my unread emails"
    and "show  my unread EMAILS" share one entry — a cheap step toward semantic
    caching without an embedding lookup. The decision is user-independent (it
    only classifies the request text), so the key carries no user_id.
    """
    normalized = " ".join(user_request.lower().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]
    return f"routercache:{digest}"


async def _cached_route(user_request: str) -> _RouteDecision | None:
    """Return a previously cached router decision, or None. Fails open on Redis errors."""
    if get_settings().router_cache_ttl_sec <= 0:
        return None
    try:
        raw = await get_redis().get(_router_cache_key(user_request))
    except Exception as exc:
        log.warning("router_cache_read_failed", error=str(exc))
        return None
    if not raw:
        return None
    try:
        return _RouteDecision.model_validate_json(raw)
    except Exception:
        return None  # corrupt entry — treat as a miss, the LLM call overwrites it


async def _store_route(user_request: str, decision: _RouteDecision) -> None:
    """Best-effort write of a router decision to Redis; errors only logged."""
    ttl = get_settings().router_cache_ttl_sec
    if ttl <= 0:
        return
    try:
        await get_redis().set(_router_cache_key(user_request),
                              decision.model_dump_json(), ex=ttl)
    except Exception as exc:
        log.warning("router_cache_write_failed", error=str(exc))


async def _route_query(state: AgentState, metadata: dict[str, str]) -> _RouteDecision:
    """One structured call: decide IF retrieval helps AND rewrite the search query.

    Replaces the old two-call decide → rewrite preamble, removing one LLM
    round-trip from every request's critical path. On any LLM failure, fall
    back to retrieving with the raw request — retrieval is best-effort and
    must never be run-fatal.

    Decisions are cached in Redis keyed by the normalized request text (see
    _router_cache_key), so a repeated request skips this LLM call entirely.
    Only successful LLM decisions are cached — never the fallback.
    """
    cached = await _cached_route(state.user_request)
    if cached is not None:
        log.info("retrieve_route_cache_hit", should_retrieve=cached.should_retrieve)
        return cached
    msgs = build_retriever_router_messages(user_request=state.user_request)
    llm = get_structured_llm("retriever-grader", _RouteDecision, metadata)
    try:
        decision: _RouteDecision = await llm.ainvoke(msgs)
    except Exception as exc:
        log.warning("retrieve_route_fallback", error=str(exc))
        return _RouteDecision(should_retrieve=True, query=state.user_request)
    if not decision.query.strip():
        decision.query = state.user_request
    await _store_route(state.user_request, decision)
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
            # Kept in cosine space (not the RRF fused score) so it stays a meaningful,
            # cross-query-comparable similarity for the UI; guard.py deliberately
            # excludes it from the confidence formula.
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
    hits = (await get_qdrant().query_points(
        collection_name=_PLANS_COLLECTION,
        query=vec,
        using=DENSE_VECTOR,
        query_filter=qfilter,
        limit=k,
    )).points
    raw = [_to_plan(h, float(getattr(h, "score", 0.0)), float(getattr(h, "score", 0.0)))
           for h in hits]
    plans = [p for p in raw if p is not None]
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
    # The RRF-fused pass and the cosine-recovery dense pass are independent, so
    # run them concurrently — one Qdrant round-trip of latency instead of two.
    # (RRF drops per-vector scores, hence the separate dense pass for cosine.)
    fused_resp, dense_resp = await asyncio.gather(
        client.query_points(
            collection_name=_PLANS_COLLECTION,
            prefetch=[
                Prefetch(query=vec, using=DENSE_VECTOR, filter=qfilter, limit=k),
                Prefetch(query=sparse, using=SPARSE_VECTOR, filter=qfilter, limit=k),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
        ),
        client.query_points(
            collection_name=_PLANS_COLLECTION,
            query=vec,
            using=DENSE_VECTOR,
            query_filter=qfilter,
            limit=k,
        ),
    )
    t2 = time.monotonic()
    cosine = {str(getattr(h, "id", "")): float(getattr(h, "score", 0.0))
              for h in dense_resp.points}
    plans: list[RetrievedPlan] = []
    for hit in fused_resp.points:
        hid = str(getattr(hit, "id", ""))
        plan = _to_plan(hit, cosine.get(hid, 0.0), float(getattr(hit, "score", 0.0)))
        if plan is not None:
            plans.append(plan)
    log.info("search_timing", mode="hybrid",
             sparse_ms=int((t1 - t0) * 1000), query_ms=int((t2 - t1) * 1000),
             total_ms=int((t2 - t0) * 1000), hits=len(plans))
    return plans


async def _rerank(query: str, candidates: list[RetrievedPlan]) -> list[RetrievedPlan]:
    """Cross-encoder rerank: re-score candidates by joint (query, summary) relevance.

    A no-op (returns the input order unchanged) unless RERANK_ENABLED is set. When on,
    the cross-encoder jointly encodes the query with each candidate's summary — far more
    accurate than the bi-encoder/RRF first-stage order — and we keep the top
    ``_RERANK_TOP_N``. This is a cheap pre-filter that narrows what the LLM grader sees.
    """
    if not get_settings().rerank_enabled or not candidates:
        return candidates
    # Score against the human-readable summary, falling back to the raw request text.
    docs = [c.summary or c.request_text for c in candidates]
    scores = await rerank_scores(query, docs)
    paired = list(zip(scores, candidates))
    paired.sort(key=lambda sc: sc[0], reverse=True)
    log.info("retrieve_reranked",
             scores=[{"plan_id": c.plan_id, "score": round(s, 3)} for s, c in paired])
    return [c for _, c in paired[:_RERANK_TOP_N]]


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
    hits = await _search(route.query, {"user_id": state.user_id, "success": True})
    fetched = list(hits)  # everything retrieval pulled in, for the token log below
    candidates = await _rerank(route.query, hits)
    graded = await _grade(candidates, state.user_request, meta)
    log.info("retrieve_user_scoped", candidates=len(candidates), graded=len(graded))
    if len(graded) < _MIN_AFTER_GRADE:
        broad_hits = await _search(route.query, {"success": True})
        fetched.extend(broad_hits)
        broad = await _rerank(route.query, broad_hits)
        graded = await _grade(broad, state.user_request, meta)
        log.info("retrieve_broadened", candidates=len(broad), graded=len(graded))
    # When the cross-encoder reranker is on, its ordering is the best signal we have, so
    # keep it. Otherwise order by the RRF fused score (== cosine in dense-only mode).
    if not get_settings().rerank_enabled:
        graded.sort(key=lambda p: p.fused_score, reverse=True)
    kept = graded[:_TOP_N]
    # Pre vs post token accounting: how much text search fetched vs what the
    # rerank+grade funnel let through toward the planner. Estimates (chars/4),
    # emitted per run so the reduction is visible in app logs and dashboards.
    pre_tokens = _estimate_tokens(fetched)
    post_tokens = _estimate_tokens(kept)
    log.info("retrieve_token_counts",
             pre_retrieval_tokens=pre_tokens,
             post_retrieval_tokens=post_tokens,
             tokens_dropped=pre_tokens - post_tokens,
             fetched=len(fetched), kept=len(kept))
    return kept
