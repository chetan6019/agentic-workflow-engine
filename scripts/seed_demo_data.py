"""Seed Qdrant `plans` with 30 realistic demo plans + create the demo user."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qdrant_client.http.models import PointStruct

from app.data.db import init_db
from app.data.repositories import create_user, get_user_by_username
from app.rag.embedder import embed_named
from app.rag.qdrant_client import get_qdrant, recreate_collections
from app.security.passwords import hash_password

_DEMO_USER = "demo"
_DEMO_PASS = "demo123"
_WRITE_ACTIONS = {"send_email", "reply_thread", "send_message", "schedule_message",
                  "delete_event", "delete_page", "update_event", "update_page",
                  "create_event", "create_page", "create_draft", "append_block"}

# (request, summary, [ (server, action, arguments), ... ])
_PLANS: list[tuple[str, str, list[tuple[str, str, dict]]]] = [
    # ---- Calendar (8) ----
    ("Schedule a team standup tomorrow at 9am", "Create a standup event tomorrow 09:00–09:30.",
     [("calendar", "create_event", {"summary": "Team standup",
        "start": "2026-06-01T09:00:00Z", "end": "2026-06-01T09:30:00Z"})]),
    ("Block focus time this Friday afternoon", "Create a 3-hour focus block Fri 13:00–16:00.",
     [("calendar", "create_event", {"summary": "Focus time",
        "start": "2026-06-05T13:00:00Z", "end": "2026-06-05T16:00:00Z"})]),
    ("Find a free 1-hour slot with Alice this week", "Query freebusy for an open hour.",
     [("calendar", "find_free_slot", {"time_min": "2026-05-30T00:00:00Z",
        "time_max": "2026-06-06T00:00:00Z", "duration_minutes": 60})]),
    ("Reschedule tomorrow's 2pm meeting to 4pm", "Patch event start/end to the new time.",
     [("calendar", "update_event", {"event_id": "<id>",
        "start": "2026-06-01T16:00:00Z", "end": "2026-06-01T17:00:00Z"})]),
    ("List my meetings for the next 5 days", "Fetch upcoming events from primary calendar.",
     [("calendar", "list_upcoming", {"max_results": 20})]),
    ("Delete the canceled Q3 review event", "Remove the obsolete event by id.",
     [("calendar", "delete_event", {"event_id": "<id>"})]),
    ("Schedule a 1:1 with my manager Monday 10am", "Create a 30-min 1:1 next Monday 10:00.",
     [("calendar", "create_event", {"summary": "1:1 with manager",
        "start": "2026-06-02T10:00:00Z", "end": "2026-06-02T10:30:00Z"})]),
    ("Find my next available 30-min slot today", "Query freebusy for a 30-min slot today.",
     [("calendar", "find_free_slot", {"time_min": "2026-05-30T00:00:00Z",
        "time_max": "2026-05-30T23:59:59Z", "duration_minutes": 30})]),
    # ---- Gmail (8) ----
    ("Show my unread emails", "List Primary inbox unread.",
     [("gmail", "search_email", {"query": "in:inbox category:primary is:unread",
        "max_results": 10})]),
    ("Show LinkedIn job alerts from this week", "Filter LinkedIn unread by recency.",
     [("gmail", "search_email", {"query": "from:linkedin.com is:unread newer_than:7d",
        "max_results": 10})]),
    ("Find emails from john@example.com", "Sender-filtered search.",
     [("gmail", "search_email", {"query": "from:john@example.com", "max_results": 10})]),
    ("Send John the report", "Resolve John's email, then send.",
     [("gmail", "search_contacts", {"query": "John", "max_results": 3}),
      ("gmail", "send_email", {"to": "<email from s1>", "subject": "Report",
        "body": "Hi John, please find the report attached."})]),
    ("Draft a follow-up to the client about the proposal", "Create a Gmail draft (don't send).",
     [("gmail", "create_draft", {"to": "client@example.com", "subject": "Proposal follow-up",
        "body": "Hi — checking in on the proposal."})]),
    ("Reply to the support thread saying I'll fix it tomorrow", "Reply to an existing thread.",
     [("gmail", "reply_thread", {"thread_id": "<id>",
        "body": "Thanks — I'll have a fix out tomorrow."})]),
    ("Find Alice's email", "Look up Alice in contacts.",
     [("gmail", "search_contacts", {"query": "Alice", "max_results": 5})]),
    ("Find emails with attachments from last week", "Filter by has:attachment + recency.",
     [("gmail", "search_email", {"query": "has:attachment newer_than:7d", "max_results": 10})]),
    # ---- Notion (7) ----
    ("Create a project page for the Q3 roadmap", "Create a Notion page under workspace root.",
     [("notion", "create_page", {"parent_id": "<workspace>", "title": "Q3 Roadmap",
        "content": "## Goals\n- ...\n## Milestones\n- ..."})]),
    ("Add today's standup notes to the team wiki", "Append a paragraph block to the wiki page.",
     [("notion", "append_block", {"block_id": "<page>", "text": "Standup: shipped X, blocked on Y."})]),
    ("Search Notion for onboarding pages", "Notion semantic search.",
     [("notion", "search_pages", {"query": "onboarding", "max_results": 10})]),
    ("Mark the product spec page as reviewed", "Update page properties.",
     [("notion", "update_page", {"page_id": "<id>", "title": "Product Spec [Reviewed]"})]),
    ("Archive my old notes page", "Set archived flag.",
     [("notion", "update_page", {"page_id": "<id>", "archived": True})]),
    ("Create a weekly review page", "New personal page with template.",
     [("notion", "create_page", {"parent_id": "<workspace>", "title": "Weekly Review",
        "content": "## Wins\n## Misses\n## Next week"})]),
    ("Find all Notion pages about API design", "Notion search by topic.",
     [("notion", "search_pages", {"query": "API design", "max_results": 10})]),
    # ---- Slack (7) ----
    ("Post a reminder to #engineering that release is tomorrow", "Send a message to a channel.",
     [("slack", "send_message", {"channel": "#engineering",
        "text": "Reminder: release ships tomorrow EOD."})]),
    ("Schedule a Monday standup reminder to #team at 8:55am", "Schedule a future message.",
     [("slack", "schedule_message", {"channel": "#team",
        "text": "Standup in 5 min", "post_at": 1780668900})]),
    ("Search Slack for 'payment bug'", "Keyword search across channels.",
     [("slack", "search_messages", {"query": "payment bug", "count": 20})]),
    ("List all my Slack channels", "Fetch channel list.",
     [("slack", "list_channels", {"limit": 100})]),
    ("Notify #sales that the new feature is live", "Send launch announcement.",
     [("slack", "send_message", {"channel": "#sales",
        "text": "🎉 New feature is live — see the docs in Notion."})]),
    ("Search Slack for messages about Q3 planning", "Keyword search.",
     [("slack", "search_messages", {"query": "Q3 planning", "count": 20})]),
    ("Schedule Friday 4pm weekly update to #general", "Schedule a recurring-style message.",
     [("slack", "schedule_message", {"channel": "#general",
        "text": "Weekly update inside — please read.", "post_at": 1781020800})]),
]


def _build_plan(req: str, summary: str, step_specs: list[tuple[str, str, dict]]) -> dict:
    """Build a valid ExecutionPlan-shaped dict from compact step specs."""
    steps = []
    for i, (server, action, args) in enumerate(step_specs, 1):
        steps.append({
            "id": f"s{i}", "tool": server, "action": action, "arguments": args,
            "depends_on": [f"s{i-1}"] if i > 1 else [],
        })
    needs_approval = any(a in _WRITE_ACTIONS for _, a, _ in step_specs)
    return {"reasoning": summary, "steps": steps,
            "strategy": "sequential" if len(steps) > 1 else "sequential",
            "complexity_score": min(10, max(1, len(steps) * 2)),
            "estimated_cost_usd": 0.01, "requires_approval": needs_approval}


async def main() -> None:
    """Create demo user, ensure Qdrant collections, and seed 30 plans."""
    await init_db()
    # Drop + recreate so the named dense+sparse schema replaces any old single-vector
    # collections left over from before hybrid search.
    await recreate_collections(["plans", "preferences"])

    existing = await get_user_by_username(_DEMO_USER)
    user_id = (existing["id"] if existing
               else (await create_user(_DEMO_USER, hash_password(_DEMO_PASS)))["id"])
    print(f"Demo user: {user_id}")

    client = get_qdrant()
    points: list[PointStruct] = []
    for req, summary, step_specs in _PLANS:
        plan_json = _build_plan(req, summary, step_specs)
        text = f"{req} {summary}"
        vec = await embed_named(text)
        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"demo-plan::{req}"))
        points.append(PointStruct(id=pid, vector=vec, payload={
            "plan_id": pid, "request_text": req, "summary": summary,
            "plan_json": plan_json, "user_id": user_id, "success": True,
        }))
    await client.upsert(collection_name="plans", points=points)
    print(f"Seeded {len(points)} demo plans into Qdrant.")


if __name__ == "__main__":
    asyncio.run(main())
