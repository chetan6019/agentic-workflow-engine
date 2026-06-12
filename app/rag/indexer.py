"""Background indexer for plans and user preferences into Qdrant."""

from __future__ import annotations

import uuid
import structlog

from qdrant_client.http.models import PointStruct

from app.core.state import AgentState
from app.rag.embedder import embed_named
from app.rag.qdrant_client import get_qdrant

log = structlog.get_logger(__name__)


async def index_plan(state: AgentState) -> None:
    """Embed and upsert a completed workflow run into the Qdrant plans collection."""
    if not state.plan:
        log.debug("index_plan_skipped_no_plan", trace_id=state.trace_id)
        return
    text = f"{state.user_request} {state.plan.reasoning}"
    vec = await embed_named(text)
    point = PointStruct(
        id=state.trace_id,
        vector=vec,
        payload={
            "plan_id": state.trace_id,
            "request_text": state.user_request,
            "plan_json": state.plan.model_dump(),
            "summary": state.plan.reasoning[:200],
            "user_id": state.user_id,
            "success": state.error is None,
        },
    )
    get_qdrant().upsert(collection_name="plans", points=[point])
    log.info("plan_indexed", trace_id=state.trace_id)


async def index_preference(user_id: str, kind: str, text: str) -> None:
    """Embed and upsert a user preference into the Qdrant preferences collection."""
    vec = await embed_named(text)
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vec,
        payload={"user_id": user_id, "kind": kind, "text": text},
    )
    get_qdrant().upsert(collection_name="preferences", points=[point])
    log.info("preference_indexed", user_id=user_id, kind=kind)
