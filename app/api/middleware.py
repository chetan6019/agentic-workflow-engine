import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.metrics import request_duration_seconds

log = structlog.get_logger(__name__)

# Probe/scrape paths hit every ~15s by compose healthchecks and Prometheus;
# logging each would flood Loki with noise that drowns real traffic.
_QUIET_PATHS = {"/healthz", "/readyz", "/metrics"}


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Fresh context per request; never inherit from a previous one.
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=request.headers.get("x-trace-id") or str(uuid.uuid4()),
            request_id=request.headers.get("x-request-id") or str(uuid.uuid4()),
            method=request.method,
            path=request.url.path,
            user_id=getattr(request.state, "user_id", None),
            session_id=getattr(request.state, "session_id", None),
        )
        try:
            return await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()


class RequestDurationMiddleware(BaseHTTPMiddleware):
    """HTTP RED metric: observe duration per (route, method, status).

    Kept separate from RequestContextMiddleware because the two have different
    concerns: a metric observation failing should never disturb a logging concern,
    and vice versa. Register BOTH in app/main.py (order doesn't matter — they
    don't depend on each other).

    `route` uses `request.url.path` because this codebase doesn't have path
    parameters with high cardinality (UUIDs etc.) — if that ever changes, swap
    to `request.scope["route"].path` to use the path TEMPLATE
    (e.g. "/v1/runs/{id}") instead of the literal path.
    """

    async def dispatch(self, request, call_next):
        start = time.monotonic()
        status = "500"  # assume failure until proven otherwise
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            # `finally` so even uncaught exceptions still contribute a sample,
            # otherwise error latencies are silently excluded from the histogram.
            duration = time.monotonic() - start
            request_duration_seconds.labels(
                route=request.url.path,
                method=request.method,
                status=status,
            ).observe(duration)
            # One access-log line per real request (probes/scrapes excluded) so
            # Loki always has a per-request signal, not only workflow-run events.
            if request.url.path not in _QUIET_PATHS:
                log.info("http_request",
                         method=request.method,
                         path=request.url.path,
                         status=status,
                         duration_ms=int(duration * 1000))