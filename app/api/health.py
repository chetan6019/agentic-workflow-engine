"""Health and readiness check endpoints."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from app.data.db import get_engine
from app.data.redis_client import get_redis
from app.rag.qdrant_client import get_qdrant

log = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict:
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "ok"}


@router.get("/metrics")
async def metrics() -> Response:
    """Prometheus scrape endpoint exposing the default registry (guardrail counters)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/readyz")
async def readyz() -> JSONResponse:
    """Readiness probe — checks Postgres, Redis, and Qdrant connectivity."""
    errors: list[str] = []

    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        errors.append(f"postgres: {exc}")
        log.error("readyz_postgres_fail", error=str(exc))

    try:
        await get_redis().ping()
    except Exception as exc:
        errors.append(f"redis: {exc}")
        log.error("readyz_redis_fail", error=str(exc))

    try:
        await get_qdrant().get_collections()
    except Exception as exc:
        errors.append(f"qdrant: {exc}")
        log.error("readyz_qdrant_fail", error=str(exc))

    if errors:
        return JSONResponse(status_code=503, content={"status": "degraded", "errors": errors})
    return JSONResponse(status_code=200, content={"status": "ok"})
