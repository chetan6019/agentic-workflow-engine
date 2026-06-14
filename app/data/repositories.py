"""Flat async repository functions — one per DB operation, returns plain dicts."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from sqlalchemy import select, update

from app.core.state import AgentState, ToolResult
from app.data.db import get_async_session
from app.data.models import Approval, Feedback, IntegrationToken, Plan, Session, ToolCall, User

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
        log.info("session_renamed", session_id=session_id, ok=result.rowcount > 0)
        return result.rowcount > 0


async def save_plan(state: AgentState) -> None:
    """Upsert an AgentState snapshot into the plans table."""
    async with get_async_session() as s:
        existing = (await s.execute(select(Plan).where(Plan.trace_id == state.trace_id))).scalar_one_or_none()
        if existing:
            existing.state_json = state.model_dump_json()
        else:
            s.add(Plan(trace_id=state.trace_id, user_id=state.user_id, state_json=state.model_dump_json()))
        log.info("plan_saved", trace_id=state.trace_id)


async def get_plan_by_trace_id(trace_id: str) -> dict[str, Any] | None:
    """Fetch the persisted AgentState snapshot for a trace."""
    async with get_async_session() as s:
        row = (await s.execute(select(Plan).where(Plan.trace_id == trace_id))).scalar_one_or_none()
        return {"trace_id": row.trace_id, "state_json": row.state_json} if row else None


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


async def save_feedback(trace_id: str, score: int, comment: str | None = None) -> None:
    """Persist user feedback for a completed run."""
    async with get_async_session() as s:
        s.add(Feedback(trace_id=trace_id, score=score, comment=comment))
        log.info("feedback_saved", trace_id=trace_id, score=score)


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
