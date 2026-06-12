"""Qdrant client singleton and collection bootstrapping."""

from __future__ import annotations

import structlog
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    SparseVectorParams,
    VectorParams,
)

from app.config import get_settings

log = structlog.get_logger(__name__)

_COLLECTIONS = ["plans", "preferences", "tool_capability_docs"]
# Named vectors, so each point carries both a dense semantic vector and a sparse
# BM25 lexical vector for hybrid (RRF-fused) retrieval. Reference these names
# anywhere a query or upsert selects a vector.
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
# Must match the embedding model's output dimension.
# bge-base-en-v1.5 → 768.  OpenAI text-embedding-3-small → 1536.
# Switching models requires dropping and re-creating all collections.
_DIM = 768
_DISTANCE = Distance.COSINE
_HNSW = HnswConfigDiff(m=16)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    """Return the cached Qdrant client."""
    url = get_settings().qdrant_url
    log.debug("qdrant_client_created", url=url)
    return QdrantClient(url=url)


def ensure_collections() -> None:
    """Idempotently create all required Qdrant collections if they don't exist."""
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    for name in _COLLECTIONS:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE_VECTOR: VectorParams(
                        size=_DIM, distance=_DISTANCE, hnsw_config=_HNSW
                    )
                },
                sparse_vectors_config={SPARSE_VECTOR: SparseVectorParams()},
            )
            log.info("qdrant_collection_created", name=name)
        else:
            log.debug("qdrant_collection_exists", name=name)


def recreate_collections(names: list[str] | None = None) -> None:
    """Drop the given collections (default: all) and recreate them with the current schema.

    Destructive: existing points are lost. Use after a schema change — e.g. the switch
    to named dense+sparse vectors — since ``ensure_collections`` leaves existing
    collections untouched and so cannot migrate an old single-vector layout.
    """
    client = get_qdrant()
    for name in names or _COLLECTIONS:
        try:
            client.delete_collection(name)
            log.info("qdrant_collection_dropped", name=name)
        except Exception as exc:
            log.debug("qdrant_collection_drop_skipped", name=name, error=str(exc))
    ensure_collections()
