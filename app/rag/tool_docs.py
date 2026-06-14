"""Vector search over the tool_capability_docs collection.

One helper shared by the planner (which tools could fulfil this request?) and
the composer's direct-answer path (what can I honestly claim to do?), replacing
two near-identical private copies.
"""

from __future__ import annotations

import structlog

from app.core.state import ToolSpec
from app.rag.embedder import embed_text
from app.rag.qdrant_client import DENSE_VECTOR, get_qdrant

log = structlog.get_logger(__name__)

_TOOL_DOC_COLLECTION = "tool_capability_docs"


async def fetch_tool_specs(query: str, k: int) -> list[ToolSpec]:
    """Return the top-k ToolSpecs most relevant to query; [] on any search failure.

    Best-effort: callers degrade gracefully (the planner still plans, the
    direct-answer path still answers) when the catalog is empty or unreachable.
    """
    try:
        vec = await embed_text(query)
        hits = (await get_qdrant().query_points(
            collection_name=_TOOL_DOC_COLLECTION, query=vec, using=DENSE_VECTOR, limit=k
        )).points
    except Exception as exc:
        log.warning("tool_specs_fetch_failed", error=str(exc))
        return []
    specs: list[ToolSpec] = []
    for hit in hits:
        payload = getattr(hit, "payload", None) or {}
        try:
            specs.append(ToolSpec.model_validate(payload))
        except Exception:
            continue
    log.debug("tool_specs_fetched", candidates=len(hits), parsed=len(specs))
    return specs
