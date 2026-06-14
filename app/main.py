"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api import approvals, auth, feedback, health, integrations, invoke, sessions
from app.config import get_settings
from app.data.db import init_db
from app.logging import configure_logging
from app.security.jwt_tokens import decode_access_token

log = structlog.get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create DB schema before serving; a failure here aborts startup loudly."""
    await init_db()
    log.info("app_started")
    yield


async def _jwt_middleware(request: Request, call_next) -> Response:
    """Extract the bearer token, decode it, and attach user_id to request.state."""
    public = {"/healthz", "/readyz", "/v1/auth/register", "/v1/auth/login"}
    if request.url.path not in public and not request.url.path.startswith("/v1/approvals"):
        auth_header = request.headers.get("Authorization", "")
        token = auth_header.removeprefix("Bearer ") if auth_header.startswith("Bearer ") else ""
        request.state.user_id = decode_access_token(token) if token else None
    return await call_next(request)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    configure_logging()
    s = get_settings()

    app = FastAPI(title="Workflow Agent API", version="1.0.0", lifespan=_lifespan)

    app.middleware("http")(_jwt_middleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[s.streamlit_origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in [auth.router, invoke.router, sessions.router, approvals.router,
                   feedback.router, integrations.router, health.router]:
        app.include_router(router)

    return app


app = create_app()
