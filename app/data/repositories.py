"""Flat async repository functions — one per DB operation, returns plain dicts."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from sqlalchemy import delete, select, update

from app.core.state import AgentState, ToolResult
from app.data.db import get_async_session
from app.data.models import (
    Approval,
    AuditLog,
    Feedback,
    IntegrationToken,
    Message,
    Plan,
    Session,
    ToolCall,
    User,
)

log = structlog.get_logger(__name__)


async def create_user(username: str, password_hash: str) -> dict[str, Any]:
    """Insert a new user row and return its id + username."""
    async with get_async_session() as s:
        user = User(username=username, password_hash=password_hash)
        s.add(user)
        await s.flush()
        log.info("user_created", user_id=user.id)
        return {"id": user.id, "username": user.username}


async def get_user_by_username(username: str) -> dict[str, Any] | None:
    """Fetch a user row by username or return None."""
    async with get_async_session() as s:
        row = (await s.execute(select(User).where(User.username == username))).scalar_one_or_none()
        return {"id": row.id, "username": row.username, "password_hash": row.password_hash} if row else None


async def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    """Fetch a user row by id or return None."""
    async with get_async_session() as s:
        row = (await s.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        return {"id": row.id, "username": row.username} if row else None


async def create_session(user_id: str) -> str:
    """Create a new chat session and return its id."""
    async with get_async_session() as s:
        sess = Session(user_id=user_id)
        s.add(sess)
        await s.flush()
        log.info("session_created", session_id=sess.id, user_id=user_id)
        return sess.id


async def get_sessions_by_user(user_id: str) -> list[dict[str, Any]]:
    """Return all sessions for a user as a list of dicts."""
    async with get_async_session() as s:
        rows = (await s.execute(select(Session).where(Session.user_id == user_id))).scalars().all()
        return [{"id": r.id, "title": r.title, "status": r.status, "started_at": str(r.started_at), "last_activity": str(r.last_activity)} for r in rows]


async def get_session(session_id: str) -> dict[str, Any] | None:
    """Fetch a session row by id or return None."""
    async with get_async_session() as s:
        row = (await s.execute(select(Session).where(Session.id == session_id))).scalar_one_or_none()
        return {"id": row.id, "user_id": row.user_id, "title": row.title, "status": row.status} if row else None


async def rename_session(session_id: str, user_id: str, title: str) -> bool:
    """Set a session's title (only if it belongs to the user). Returns True on success."""
    async with get_async_session() as s:
        result = await s.execute(
            update(Session).where(Session.id == session_id, Session.user_id == user_id).values(title=title)
        )
        rowcount: int = getattr(result, "rowcount", 0) or 0
        renamed = rowcount > 0
        log.info("session_renamed", session_id=session_id, ok=renamed)
        return renamed


async def save_message(session_id: str, role: str, content: str) -> None:
    """Append one chat message (role: user|assistant) to a session's history."""
    async with get_async_session() as s:
        s.add(Message(session_id=session_id, role=role, content=content))
        log.debug("message_saved", session_id=session_id, role=role)


async def get_recent_messages(session_id: str, limit: int) -> list[dict[str, str]]:
    """Return the last `limit` messages for a session, oldest-first ({role, content})."""
    async with get_async_session() as s:
        rows = (await s.execute(
            select(Message).where(Message.session_id == session_id)
            .order_by(Message.created_at.desc()).limit(limit)
        )).scalars().all()
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


# Long tool outputs (e.g. full Gmail message bodies) bloat the forever-persisted
# plans row; clip any string past this length in the snapshot (REVIEW.md R36). The
# in-memory state keeps the full values — only the persisted copy is trimmed.
_PERSIST_STR_LIMIT = 2000


def _clip(value: Any) -> Any:
    """Recursively clip over-long strings inside a tool output payload."""
    if isinstance(value, str):
        return value[:_PERSIST_STR_LIMIT] + "…[truncated]" if len(value) > _PERSIST_STR_LIMIT else value
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v) for v in value]
    return value


def _persistable_json(state: AgentState) -> str:
    """Serialize state for the plans row with long tool outputs trimmed."""
    data = state.model_dump()
    for r in data.get("tool_results", []):
        if isinstance(r.get("output"), dict):
            r["output"] = _clip(r["output"])
    return json.dumps(data, default=str)


async def save_plan(state: AgentState) -> None:
    """Upsert an AgentState snapshot into the plans table (tool outputs clipped)."""
    payload = _persistable_json(state)
    async with get_async_session() as s:
        existing = (await s.execute(select(Plan).where(Plan.trace_id == state.trace_id))).scalar_one_or_none()
        if existing:
            existing.state_json = payload
        else:
            s.add(Plan(trace_id=state.trace_id, user_id=state.user_id, state_json=payload))
        log.info("plan_saved", trace_id=state.trace_id)


async def get_plan_by_trace_id(trace_id: str) -> dict[str, Any] | None:
    """Fetch the persisted AgentState snapshot for a trace (with its owner user_id)."""
    async with get_async_session() as s:
        row = (await s.execute(select(Plan).where(Plan.trace_id == trace_id))).scalar_one_or_none()
        return {"trace_id": row.trace_id, "user_id": row.user_id,
                "state_json": row.state_json} if row else None


async def save_tool_call(trace_id: str, result: ToolResult) -> None:
    """Persist a single ToolResult into the tool_calls table."""
    async with get_async_session() as s:
        s.add(ToolCall(trace_id=trace_id, step_id=result.step_id, ok=result.ok, latency_ms=result.latency_ms, error=result.error))
        log.debug("tool_call_saved", trace_id=trace_id, step_id=result.step_id, ok=result.ok)


async def create_approval(trace_id: str, hours: int = 24) -> str:
    """Create a HITL approval token expiring in `hours` hours; return token."""
    async with get_async_session() as s:
        token = str(uuid.uuid4())
        expires = datetime.now(timezone.utc) + timedelta(hours=hours)
        s.add(Approval(token=token, trace_id=trace_id, expires_at=expires))
        log.info("approval_created", trace_id=trace_id, token=token[:8] + "…")
        return token


async def get_or_create_approval(trace_id: str, hours: int = 24) -> str:
    """Return the existing UNDECIDED approval token for this trace, or create one.

    interrupt() replays the paused node from the top on resume, so the pause path
    runs twice per approval — this must never mint a second token for the same run.
    """
    async with get_async_session() as s:
        row = (await s.execute(
            select(Approval).where(Approval.trace_id == trace_id,
                                   Approval.decision.is_(None))
        )).scalars().first()
        if row:
            return row.token
    return await create_approval(trace_id, hours)


async def get_approval_by_token(token: str) -> dict[str, Any] | None:
    """Fetch approval row by token or return None."""
    async with get_async_session() as s:
        row = (await s.execute(select(Approval).where(Approval.token == token))).scalar_one_or_none()
        return {"token": row.token, "trace_id": row.trace_id, "decision": row.decision, "expires_at": row.expires_at} if row else None


async def update_approval_status(token: str, decision: str) -> None:
    """Record the user's decision on an approval row."""
    async with get_async_session() as s:
        await s.execute(update(Approval).where(Approval.token == token).values(decision=decision))
        log.info("approval_updated", token=token[:8] + "…", decision=decision)


async def get_pending_approvals_by_user(user_id: str) -> list[dict[str, Any]]:
    """Return undecided approvals for the user's own runs (joined via plans.user_id)."""
    async with get_async_session() as s:
        rows = (await s.execute(
            select(Approval).join(Plan, Approval.trace_id == Plan.trace_id)
            .where(Plan.user_id == user_id, Approval.decision.is_(None))
        )).scalars().all()
        return [{"token": r.token, "trace_id": r.trace_id,
                 "expires_at": r.expires_at, "decision": r.decision} for r in rows]


async def save_feedback(trace_id: str, score: int, comment: str | None = None) -> None:
    """Persist user feedback for a completed run."""
    async with get_async_session() as s:
        s.add(Feedback(trace_id=trace_id, score=score, comment=comment))
        log.info("feedback_saved", trace_id=trace_id, score=score)


async def save_audit(user_id: str, action: str, detail: str | None = None) -> None:
    """Append an immutable audit-trail entry (used by guardrail blocks)."""
    async with get_async_session() as s:
        s.add(AuditLog(user_id=user_id, action=action, detail=detail))
        log.info("audit_logged", user_id=user_id, action=action)


async def save_token(user_id: str, provider: str, token_enc: str) -> None:
    """Upsert an encrypted integration token for a user + provider pair."""
    async with get_async_session() as s:
        existing = (await s.execute(select(IntegrationToken).where(IntegrationToken.user_id == user_id, IntegrationToken.provider == provider))).scalar_one_or_none()
        if existing:
            existing.token_enc = token_enc
        else:
            s.add(IntegrationToken(user_id=user_id, provider=provider, token_enc=token_enc))
        log.info("token_saved", user_id=user_id, provider=provider)


async def get_token(user_id: str, server: str) -> str | None:
    """Fetch the encrypted integration token for a user + server/provider."""
    async with get_async_session() as s:
        row = (await s.execute(select(IntegrationToken).where(IntegrationToken.user_id == user_id, IntegrationToken.provider == server))).scalar_one_or_none()
        return row.token_enc if row else None


async def get_user_providers(user_id: str) -> list[str]:
    """Return the providers a user has stored an integration token for."""
    async with get_async_session() as s:
        rows = (await s.execute(select(IntegrationToken.provider).where(IntegrationToken.user_id == user_id))).scalars().all()
        return list(rows)


async def get_user_tokens(user_id: str) -> list[dict[str, Any]]:
    """Return the user's integration rows as {provider, token_enc} for status checks."""
    async with get_async_session() as s:
        rows = (await s.execute(
            select(IntegrationToken).where(IntegrationToken.user_id == user_id))).scalars().all()
        return [{"provider": r.provider, "token_enc": r.token_enc} for r in rows]


async def delete_token(user_id: str, provider: str) -> bool:
    """Delete a user's provider token row; return True if a row was removed."""
    async with get_async_session() as s:
        result = await s.execute(delete(IntegrationToken).where(
            IntegrationToken.user_id == user_id, IntegrationToken.provider == provider))
        removed = (getattr(result, "rowcount", 0) or 0) > 0
        log.info("token_deleted", user_id=user_id, provider=provider, removed=removed)
        return removed
