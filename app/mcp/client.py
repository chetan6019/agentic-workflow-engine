"""MCP client wrapping `langchain-mcp-adapters` MultiServerMCPClient."""

from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from typing import Any

import httpx
import structlog
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from opentelemetry import trace
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
# mcp_tool_latency_seconds removed — tool latency now comes from the OTel tool span.
from app.core.state import ToolResult, ToolSpec
from app.data.redis_client import get_redis

log = structlog.get_logger(__name__)
# No-op tracer unless app/core/tracing.py configured a provider at startup.
_tracer = trace.get_tracer(__name__)
_CACHE_TTL_SEC = 900
_TRANSIENT_EXC: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    ConnectionError,
    TimeoutError,
)


def _idempotency_key(trace_id: str, step_id: str, tool: str, args: dict[str, Any]) -> str:
    """Hash trace_id + step_id + tool + sorted args into a stable idempotency key."""
    payload = json.dumps(args, sort_keys=True, default=str)
    raw = f"{trace_id}:{step_id}:{tool}:{payload}".encode()
    return hashlib.sha256(raw).hexdigest()


def _parse_tool_output(raw: Any) -> dict[str, Any]:
    """Unwrap and parse the adapter's return value into the tool's structured dict.

    MCP servers return JSON, but the transport can wrap it: the adapter may hand
    us the dict itself, a JSON string, or a content-block envelope like
    [{"type": "text", "text": "<json>"}]. Parse down to real fields — if the
    result stays one serialized string, prompt compaction later clips it at
    _RESULT_STR_LIMIT chars and silently loses items (an email list showed only
    its first entry to the composer). Non-JSON values keep the {"value": ...}
    wrap so ToolResult.output is always a dict.
    """
    # A single text content block → unwrap to its inner text, then fall through.
    if (isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict)
            and raw[0].get("type") == "text"):
        raw = raw[0].get("text", "")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {"value": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    if isinstance(raw, dict):
        return raw
    return {"value": raw}


def _to_tool_result(step_id: str, raw: Any, started: float, error: str | None = None) -> ToolResult:
    """Normalise an adapter return value (or error) into a ToolResult."""
    latency = int((time.monotonic() - started) * 1000)
    if error is not None:
        return ToolResult(step_id=step_id, ok=False, output=None, error=error, latency_ms=latency)
    return ToolResult(step_id=step_id, ok=True, output=_parse_tool_output(raw),
                      error=None, latency_ms=latency)


class MCPClient:
    """Project-local wrapper around langchain-mcp-adapters MultiServerMCPClient."""

    def __init__(self, server_config: dict[str, dict[str, Any]]) -> None:
        """Build the adapter client and prepare the lazy tool registry."""
        # Pre-existing: the adapter accepts plain connection dicts at runtime, but its type hint
        # wants typed *Connection unions; ignore the arg-type mismatch (not introduced by this PR).
        self._mcp = MultiServerMCPClient(server_config)  # type: ignore[arg-type]
        self._servers = set(server_config)
        self._tools_by_server: dict[str, dict[str, BaseTool]] = {}

    async def _load_server_tools(self, server: str) -> dict[str, BaseTool]:
        """Fetch tools for one server through the adapter and cache them."""
        if server not in self._tools_by_server:
            tools = await self._mcp.get_tools(server_name=server)
            self._tools_by_server[server] = {t.name: t for t in tools}
        return self._tools_by_server[server]

    async def _resolve(self, server: str, tool: str) -> tuple[str, str]:
        """Tolerate a planner that swaps server/action or names the action as server."""
        if server in self._servers:
            return server, tool
        if tool in self._servers:  # fields swapped
            return tool, server
        for name in (server, tool):  # find which server actually exposes the tool
            for srv in self._servers:
                try:
                    if name in await self._load_server_tools(srv):
                        return srv, name
                except Exception:
                    continue
        return server, tool

    async def call_tool(self, server: str, tool: str, args: dict[str, Any],
                        *, trace_id: str = "", step_id: str = "") -> ToolResult:
        """Invoke an MCP tool with Redis-backed idempotency and tenacity retries."""
        server, tool = await self._resolve(server, tool)
        key = _idempotency_key(trace_id, step_id, tool, args)
        redis = get_redis()
        cached = await redis.get(f"mcp:{key}")
        if cached:
            log.info("mcp_tool_cache_hit", server=server, tool=tool, step=step_id)
            return ToolResult.model_validate_json(cached)

        try:
            registry = await self._load_server_tools(server)
        except Exception as exc:
            log.warning("mcp_server_unknown", server=server, tool=tool, error=str(exc))
            return _to_tool_result(step_id, None, time.monotonic(), error=f"unknown_server:{server}")
        tool_obj = registry.get(tool)
        if tool_obj is None:
            log.warning("mcp_tool_unknown", server=server, tool=tool, step=step_id)
            return _to_tool_result(step_id, None, time.monotonic(), error=f"unknown_tool:{server}/{tool}")

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=0.5, max=4),
            retry=retry_if_exception_type(_TRANSIENT_EXC),
            reraise=True,
        )
        async def _invoke() -> Any:
            return await tool_obj.ainvoke(args)

        log.info("mcp_tool_call_start", server=server, tool=tool, step=step_id)
        started = time.monotonic()
        # Span attributes carry identifiers only (never tool args/outputs — may hold PII).
        with _tracer.start_as_current_span("mcp.tool_call") as span:
            span.set_attribute("mcp.server", server)
            span.set_attribute("mcp.tool", tool)
            try:
                raw = await _invoke()
                result = _to_tool_result(step_id, raw, started)
            except Exception as exc:
                log.warning("mcp_tool_call_failed", server=server, tool=tool,
                            step=step_id, error=str(exc))
                result = _to_tool_result(step_id, None, started, error=str(exc))
            span.set_attribute("mcp.ok", result.ok)
            # getattr: tenacity attaches .statistics to the wrapped function at runtime,
            # but its type stubs don't declare it.
            stats = getattr(_invoke, "statistics", {})
            span.set_attribute("mcp.retries", stats.get("attempt_number", 1) - 1)

        await redis.set(f"mcp:{key}", result.model_dump_json(), ex=_CACHE_TTL_SEC)
        log.info("mcp_tool_call_done", server=server, tool=tool, step=step_id,
                 ok=result.ok, latency_ms=result.latency_ms)
        return result

    async def list_tools(self, server: str) -> list[ToolSpec]:
        """Return ToolSpec entries for every tool the server exposes."""
        registry = await self._load_server_tools(server)
        specs: list[ToolSpec] = []
        for t in registry.values():
            specs.append(
                ToolSpec(name=t.name, description=t.description or "",
                         server=server, input_schema=_tool_schema(t))
            )
        return specs


def _tool_schema(tool: BaseTool) -> dict[str, Any]:
    """Extract a tool's JSON input schema, tolerant of dict or Pydantic args_schema."""
    schema = getattr(tool, "args_schema", None)
    if isinstance(schema, dict):
        return schema
    if schema is not None and hasattr(schema, "model_json_schema"):
        try:
            return schema.model_json_schema()
        except Exception:
            pass
    args = getattr(tool, "args", None)  # langchain fallback: {prop: subschema}
    return {"type": "object", "properties": args} if isinstance(args, dict) else {}


@lru_cache(maxsize=1)
def get_mcp_client() -> MCPClient:
    """Build the MCPClient singleton from configured MCP server URLs."""
    s = get_settings()

    def _ep(base: str) -> dict[str, str]:
        return {"url": f"{base.rstrip('/')}/mcp", "transport": "streamable_http"}

    server_config = {
        # STALE (2026-06-22): the four old server keys are superseded by the google/github/
        # reddit/finnhub entries below; commented (not deleted) per CLAUDE.md R13.
        # "calendar": _ep(s.mcp_calendar_url),
        # "gmail": _ep(s.mcp_gmail_url),
        # "notion": _ep(s.mcp_notion_url),
        # "slack": _ep(s.mcp_slack_url),
        "google": _ep(s.mcp_google_url),
        "github": _ep(s.mcp_github_url),
        "reddit": _ep(s.mcp_reddit_url),
        "finnhub": _ep(s.mcp_finnhub_url),
    }
    log.info("mcp_client_configured", servers=list(server_config))
    return MCPClient(server_config)
