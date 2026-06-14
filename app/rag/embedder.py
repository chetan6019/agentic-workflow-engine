"""Embed text via the LiteLLM gateway (default) or local HuggingFace, with a Redis cache.

Two providers are supported, selected by ``EMBEDDING_PROVIDER``:

- ``openai`` (default) — uses LangChain's ``OpenAIEmbeddings`` client to speak
  the OpenAI-compatible HTTP API to our LiteLLM proxy.  The label refers to
  the *client shape*, NOT the upstream model: LiteLLM decides what actually
  serves the call (today HuggingFace ``BAAI/bge-base-en-v1.5`` via the HF
  Inference API with a local TEI fallback; swap to a different upstream by
  editing ``litellm_config.yaml`` — no code change here).
- ``hf`` — uses LangChain's ``HuggingFaceEmbeddings`` to run
  sentence-transformers in this Python process.  Offline-capable, zero network,
  but bypasses LiteLLM so it loses observability, central rate limits, and the
  swap-from-config feature.

Cache keys are namespaced by provider so vectors from the two clients never
mix in Redis — dimensions and vector spaces can differ between upstreams.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from functools import lru_cache
from typing import Any, Literal

import structlog
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from qdrant_client.http.models import SparseVector

from app.config import get_settings
from app.data.redis_client import get_redis
from app.rag.qdrant_client import DENSE_VECTOR, SPARSE_VECTOR

log = structlog.get_logger(__name__)

Provider = Literal["openai", "hf"]
_CACHE_TTL = 86_400
# BM25 lexical model. Pure CPU, no network at query time after the one-off model
# download, and not an LLM embedding — keeps the LiteLLM-only rule intact.
_BM25_MODEL = "Qdrant/bm25"


def _resolved_provider() -> Provider:
    """Return the configured embedding provider (declared in Settings)."""
    return get_settings().embedding_provider


def _build_embeddings(provider: Provider) -> Embeddings:
    """Construct a LangChain embeddings client for the given provider."""
    s = get_settings()
    if provider == "hf":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=hf needs 'langchain-huggingface' and "
                "'sentence-transformers' installed (uncomment them in "
                "requirements.txt). The default 'openai' provider uses LiteLLM "
                "and needs neither."
            ) from exc
        return HuggingFaceEmbeddings(
            model_name=s.hf_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    return OpenAIEmbeddings(
        model="embed",
        openai_api_base=s.litellm_url,
        openai_api_key=s.litellm_virtual_key,
    )


def _cache_key(provider: Provider, text: str) -> str:
    """Redis key namespaced by provider so vectors from different clients never mix."""
    digest = hashlib.sha256(text.strip().lower().encode()).hexdigest()
    return f"emb:{provider}:{digest}"


async def embed_text(text: str) -> list[float]:
    """Return the embedding vector for text, served from Redis cache when available."""
    provider = _resolved_provider()
    key = _cache_key(provider, text)
    redis = get_redis()
    cached = await redis.get(key)
    if cached:
        log.debug("embed_cache_hit", key=key[:24], provider=provider)
        return json.loads(cached)
    vec: list[float] = await _build_embeddings(provider).aembed_query(text)
    await redis.set(key, json.dumps(vec), ex=_CACHE_TTL)
    log.debug("embed_cache_miss", key=key[:24], dims=len(vec), provider=provider)
    return vec


@lru_cache(maxsize=1)
def _bm25() -> Any:
    """Lazily build the cached BM25 sparse encoder (downloads the model on first use)."""
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=_BM25_MODEL)


async def embed_sparse(text: str) -> SparseVector:
    """Return the BM25 sparse vector for text, encoded off the event loop."""
    emb = await asyncio.to_thread(lambda: next(iter(_bm25().embed([text]))))
    return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())


async def embed_named(text: str) -> dict[str, Any]:
    """Return both named vectors {dense, sparse} for a hybrid-search Qdrant point."""
    dense = await embed_text(text)
    sparse = await embed_sparse(text)
    return {DENSE_VECTOR: dense, SPARSE_VECTOR: sparse}
