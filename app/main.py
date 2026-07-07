"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import approvals, auth, feedback, health, integrations, invoke, sessions
from app.api.middleware import RequestContextMiddleware, RequestDurationMiddleware
from app.config import get_settings
from app.core.background import drain, spawn
from app.data.db import get_engine, init_db
from app.data.redis_client import check_rate_limit, get_redis
from app.logging import configure_logging
from app.rag.embedder import warmup_embedder
from app.rag.qdrant_client import get_qdrant
from app.security.jwt_tokens import decode_access_token

log = structlog.get_logger(__name__)

# Liveness/readiness/metrics endpoints are never rate limited or auth-checked.
_PROBES = {"/healthz", "/readyz", "/metrics"}
# Paths that don't require a decoded JWT (auth bootstrap + token-based approvals).
_NO_JWT = _PROBES | {"/v1/auth/register", "/v1/auth/login"}
# Seconds to let in-flight background runs finish on SIGTERM before cancelling.
_SHUTDOWN_GRACE_SEC = 10


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create DB schema, then serve immediately while warming models in the background."""
    configure_logging()
    await init_db()
    # Warm the in-process embedding models OFF the boot path so the api comes up
    # instantly. They load concurrently in the background; the only cost is that a
    # query arriving in the first few seconds may still pay the cold-start once.
    spawn(_warmup())
    log.info("app_started")
    yield
    await _shutdown()


async def _warmup() -> None:
    """Background model warmup; a failure only means a slow first request, not a crash."""
    try:
        await warmup_embedder()
        log.info("embedder_warmed")
    except Exception as exc:
        log.warning("embedder_warmup_failed", error=str(exc))


async def _shutdown() -> None:
    """Drain in-flight background runs, then close DB / Redis / Qdrant cleanly."""
    await drain(_SHUTDOWN_GRACE_SEC)
    for name, closer in (("engine", get_engine().dispose),
                         ("redis", get_redis().aclose),
                         ("qdrant", get_qdrant().close)):
        try:
            await closer()
        except Exception as exc:
            log.warning("shutdown_close_failed", resource=name, error=str(exc))
    log.info("app_stopped")


async def _auth_and_rate_limit(request: Request, call_next) -> Response:
    """Decode the bearer token onto request.state, then enforce a per-identity rate limit."""
    path = request.url.path
    # Approvals decode the JWT when present (so list + ownership checks work) but do
    # not require it — the {token} capability still authorizes the POST on its own.
    if path not in _NO_JWT:
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ") if auth_header.startswith("Bearer ") else ""
        request.state.user_id = decode_access_token(token) if token else None

    # Rate limit everything except the probes — by user_id when known, else client IP
    # (so login/register brute force is throttled too).
    if path not in _PROBES:
        identity = getattr(request.state, "user_id", None) or _client_ip(request)
        if not await check_rate_limit(identity, get_settings().rate_limit_per_min):
            log.warning("rate_limited", identity=identity, path=path)
            return JSONResponse(status_code=429, content={"detail": "rate_limit_exceeded"})
    return await call_next(request)


def _client_ip(request: Request) -> str:
    """Best-effort client IP for unauthenticated rate-limit bucketing."""
    return request.client.host if request.client else "anon"


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    s = get_settings()

    app = FastAPI(title="Workflow Agent API", version="1.0.0", lifespan=_lifespan)

    app.middleware("http")(_auth_and_rate_limit)

    app.add_middleware(RequestContextMiddleware)
    # HTTP RED metric — registered AFTER RequestContextMiddleware in source order
    # so it runs OUTERMOST and times the full request including auth + rate-limit
    # (Starlette runs middleware outermost-first; last added is outermost).
    app.add_middleware(RequestDurationMiddleware)

    # Streamlit origin plus the React web UI origins (comma-separated setting).
    # In prod the React app is same-origin behind its nginx proxy, so CORS mostly
    # matters for the Vite dev server hitting :8000 directly.
    origins = [s.streamlit_origin] + [o.strip() for o in s.web_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in [auth.router, invoke.router, sessions.router, approvals.router,
                   feedback.router, integrations.router, health.router]:
        app.include_router(router)

    return app


app = create_app()
