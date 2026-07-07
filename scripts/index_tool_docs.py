"""Index MCP tool names, descriptions, and input schemas into Qdrant.

Discovers tools by connecting to the running MCP servers through the same
client the orchestrator uses, so the indexed schemas match runtime exactly.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import structlog
from qdrant_client.http.models import PointStruct

from app.mcp.client import get_mcp_client
from app.rag.embedder import embed_named
from app.rag.qdrant_client import ensure_collections, get_qdrant

log = structlog.get_logger(__name__)
# STALE (2026-06-23): old MCP servers superseded by google/github/reddit/finnhub. This seeder
# populates the tool_capability_docs collection the planner reads, so it MUST list the live
# servers — otherwise the planner keeps emitting retired tool names. Old list kept per R13.
# _SERVERS = ["calendar", "gmail", "notion", "slack"]
_SERVERS = ["finnhub", "reddit", "github", "google"]


def _strip_user_id(schema: dict) -> dict:
    """Drop user_id — the orchestrator injects it; the planner shouldn't set it."""
    props = schema.get("properties")
    if isinstance(props, dict):
        props.pop("user_id", None)
    required = schema.get("required")
    if isinstance(required, list) and "user_id" in required:
        required.remove("user_id")
    return schema


async def main() -> None:
    """List tools from every MCP server and upsert their docs into Qdrant."""
    qdrant = get_qdrant()
    # Start clean so stale/junk entries from earlier runs are removed.
    try:
        await qdrant.delete_collection("tool_capability_docs")
        log.info("tool_docs_collection_cleared")
    except Exception as exc:
        log.debug("tool_docs_collection_clear_skipped", error=str(exc))
    await ensure_collections()
    mcp = get_mcp_client()
    points: list[PointStruct] = []

    for server in _SERVERS:
        try:
            specs = await mcp.list_tools(server)
        except Exception as exc:
            log.error("tool_docs_list_failed", server=server, error=str(exc))
            continue
        for spec in specs:
            payload = {
                "name": spec.name,
                "description": spec.description,
                "server": server,
                "input_schema": _strip_user_id(dict(spec.input_schema)),
            }
            vec = await embed_named(f"{spec.name}: {spec.description}")
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{server}.{spec.name}"))
            points.append(PointStruct(id=point_id, vector=vec, payload=payload))
            log.debug("tool_doc_indexed", name=spec.name, server=server)

    if points:
        await qdrant.upsert(collection_name="tool_capability_docs", points=points)
    print(f"Indexed {len(points)} tool docs into Qdrant.")


if __name__ == "__main__":
    asyncio.run(main())
