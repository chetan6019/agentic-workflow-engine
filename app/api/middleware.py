import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware

# request_duration_seconds removed — HTTP latency now comes from OTel FastAPI spans.

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
    """Access log per request. The RED latency histogram this middleware used to
    observe was removed — HTTP durations now come from OTel FastAPI spans.
    """

    async def dispatch(self, request, call_next):
        start = time.monotonic()
        status = "500"  # assume failure until proven otherwise
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            # `finally` so even uncaught exceptions still produce an access-log line.
            duration = time.monotonic() - start
            # One access-log line per real request (probes/scrapes excluded) so
            # Loki always has a per-request signal, not only workflow-run events.
            if request.url.path not in _QUIET_PATHS:
                log.info("http_request",
                         method=request.method,
                         path=request.url.path,
                         status=status,
                         duration_ms=int(duration * 1000))