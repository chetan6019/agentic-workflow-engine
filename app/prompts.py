"""Pure prompt builder functions — no DB, LLM, or HTTP imports."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# Full email bodies are the main driver of oversized composer prompts; trim them.
_COMPOSER_BODY_LIMIT = 300

_PLANNER_SYS = """<instructions>
You are a workflow planner. Given a user request, retrieved similar plans, and available tools,
produce a typed ExecutionPlan. Use the minimum steps needed. Be explicit about dependencies.
Prefer parallel execution when steps are independent.
For EVERY step: `tool` MUST be the chosen tool's `server` value (one of: calendar, gmail,
notion, slack) and `action` MUST be that tool's `name` (e.g. send_email, search_email).
NEVER put the action name in `tool`. Put the tool's parameters in `arguments` (omit
user_id — it is injected automatically).
When a step needs a date or time (e.g. calendar start/end), interpret every relative or
clock time ("tomorrow 9am", "next Monday") relative to the current moment and timezone
given in <context>, and emit it as an RFC3339 timestamp that INCLUDES that timezone's UTC
offset (e.g. 2026-06-11T09:00:00+05:30). Never emit a bare-UTC "Z" time for a local clock time.
For any named weekday ("Monday", "coming Monday", "this Friday", "next Tuesday") you MUST
use the EXACT date listed in <weekday_reference> — never compute the weekday yourself.
"Coming"/"next"/"this <weekday>" all map to that weekday's date in the reference.
NEVER invent identifiers you were not explicitly given (event_id, message_id, page_id,
etc.). A made-up id will fail. To update or delete an EXISTING calendar event the user only
describes ("the 1:1 with Priya tomorrow"), OMIT event_id entirely and instead pass
`match_summary` (distinctive words from the event title, e.g. "Priya") plus `time_min` and
`time_max` RFC3339 timestamps bounding the day it falls on; the tool resolves the real id.
</instructions>
<output_format>
Output ONLY valid json, no prose, matching exactly:
{"reasoning": "string",
 "steps": [{"id": "string", "tool": "gmail", "action": "send_email",
            "arguments": {}, "depends_on": ["string"]}],
 "strategy": "sequential|parallel|mixed",
 "complexity_score": 5, "estimated_cost_usd": 0.0, "requires_approval": false}
Example — send an email:
{"id":"s1","tool":"gmail","action":"send_email",
 "arguments":{"to":"a@b.com","subject":"Hi","body":"Hello"},"depends_on":[]}
</output_format>"""

_RETRIEVER_REWRITER_SYS = """<instructions>
Rewrite the user query into a concise, keyword-rich search query for semantic retrieval of past workflow plans.
Output only the rewritten query string.
</instructions>"""

_RETRIEVER_GRADER_SYS = """<instructions>
You are a relevance grader for workflow plan retrieval.
When candidates is empty: decide if retrieval is needed at all (output JSON: {"should_retrieve": bool, "reason": str}).
When candidates are provided: score each candidate for relevance to the query
(output JSON: {"hits": [{"plan_id": str, "relevance": float}]}).
</instructions>"""

_GUARD_SYS = """<instructions>
You are a quality judge for AI workflow responses.
Assess tone fit, hallucination risk, and instruction adherence.
Output JSON: {"tone_fit": float, "hallucination_risk": float, "instruction_adherence": float}.
All values must be between 0.0 and 1.0.
</instructions>"""

_DIRECT_ANSWER_SYS = """<instructions>
You are the assistant for a workflow automation tool. The user has asked a
conversational or meta question that does not require running any tools —
answer it directly in plain language.

Use the tool catalog below as ground truth for what you can and cannot do. Be
honest: never claim a capability that isn't represented in the catalog.
Reference integrations by friendly name (Gmail, Google Calendar, Notion, Slack)
rather than internal tool names.
</instructions>
<output_format>
Output ONLY valid json, no prose, matching exactly:
{"summary": "string", "detail_markdown": "string", "actions_taken": [],
 "actions_pending": [], "citations": []}
`summary` is a one-line answer. `detail_markdown` is the full friendly answer
shown to the user (Markdown allowed). `actions_taken`, `actions_pending`,
`citations` must be empty lists.
</output_format>"""

_COMPOSER_SYS = """<instructions>
You are a workflow response composer. Summarise what was done clearly and honestly.
Never claim an action succeeded if its tool result shows ok=false.
Write for the END USER: never mention tools, step ids, internal names, queries, or
where the data came from. No "search_*", no citations inside the text.
`detail_markdown` is the PRIMARY user-facing answer and MUST NEVER be empty — it
should stand on its own without the summary. For a single action or confirmation
(e.g. creating/updating/deleting an event, sending a message), write a short, friendly
confirmation sentence in `detail_markdown`.
When the result is a list of items (e.g. emails), `detail_markdown` MUST be a single
Markdown TABLE — one row per item — with these columns:
  | From | Subject | Date | Summary |
Write a short, human 'Summary' from each item's snippet/body. Never put raw message
IDs in the table. Keep to one consistent format (a table) — no extra prose lists.
Write in the user's preferred style if provided.
Timestamps in the data already include their UTC offset and are in the timezone named
in <context>. Present every date/time in that timezone and never convert to, or label
times as, UTC.
</instructions>
<output_format>
Output ONLY valid json, no prose, matching exactly:
{"summary": "string", "detail_markdown": "string", "actions_taken": ["string"],
 "actions_pending": ["string"], "citations": ["string"]}
</output_format>"""


def _slim_tool_specs(tool_specs: list[Any]) -> list[dict[str, Any]]:
    """Reduce tool specs to name/description/server + param names; full JSON Schemas bloat the prompt."""
    slim: list[dict[str, Any]] = []
    for t in tool_specs:
        d = t.model_dump() if hasattr(t, "model_dump") else dict(t)
        schema = d.get("input_schema") or {}
        props = list(schema.get("properties", {})) if isinstance(schema, dict) else []
        required = schema.get("required", []) if isinstance(schema, dict) else []
        slim.append({"name": d.get("name"), "description": d.get("description"),
                     "server": d.get("server"), "params": props, "required": required})
    return slim


def _slim_examples(examples: list[Any]) -> list[dict[str, Any]]:
    """Keep only request + summary from retrieved plans; the full plan_json bloats the prompt."""
    out: list[dict[str, Any]] = []
    for e in examples[:3]:
        d = e.model_dump() if hasattr(e, "model_dump") else dict(e)
        out.append({"request": d.get("request_text"), "summary": d.get("summary")})
    return out


def build_planner_messages(user_request: str, examples: list[Any], tool_specs: list[Any],
                           now_local: str, tz_name: str, weekday_ref: str) -> list[BaseMessage]:
    """Build planner prompt with tool listing, retrieved-plan examples, and time context."""
    tools_block = json.dumps(_slim_tool_specs(tool_specs))
    examples_block = json.dumps(_slim_examples(examples))
    context = f"Current time: {now_local}\nTimezone: {tz_name}"
    human = (f"<context>\n{context}\n</context>\n\n"
             f"<weekday_reference>\n{weekday_ref}\n</weekday_reference>\n\n"
             f"<examples>\n{examples_block}\n</examples>\n\n"
             f"<tools>\n{tools_block}\n</tools>\n\n"
             f"<request>\n{user_request}\n</request>")
    return [SystemMessage(content=_PLANNER_SYS), HumanMessage(content=human)]


def build_retriever_rewriter_messages(user_request: str) -> list[BaseMessage]:
    """Build query-rewrite prompt for the retriever."""
    return [SystemMessage(content=_RETRIEVER_REWRITER_SYS),
            HumanMessage(content=f"<request>\n{user_request}\n</request>")]


def build_retriever_grader_messages(query: str, candidates: list[Any]) -> list[BaseMessage]:
    """Build grader prompt; handles empty candidates (should-retrieve) and populated lists (relevance scoring)."""
    if not candidates:
        human = f"<request>\n{query}\n</request>\n<candidates>\n[]\n</candidates>"
    else:
        cands = json.dumps([c.model_dump() if hasattr(c, "model_dump") else c for c in candidates])
        human = f"<request>\n{query}\n</request>\n<candidates>\n{cands}\n</candidates>"
    return [SystemMessage(content=_RETRIEVER_GRADER_SYS), HumanMessage(content=human)]


def build_guard_judge_messages(draft: Any, user_request: str) -> list[BaseMessage]:
    """Build guard-judge prompt from a DraftResponse and the original request."""
    draft_text = draft.model_dump_json() if hasattr(draft, "model_dump_json") else json.dumps(draft)
    human = f"<draft>\n{draft_text}\n</draft>\n\n<request>\n{user_request}\n</request>"
    return [SystemMessage(content=_GUARD_SYS), HumanMessage(content=human)]


def _compact_results(tool_results: list[Any]) -> list[Any]:
    """Trim long email bodies out of tool outputs so the composer prompt stays small."""
    compacted: list[Any] = []
    for r in tool_results:
        data = r.model_dump() if hasattr(r, "model_dump") else dict(r)
        output = data.get("output") if isinstance(data, dict) else None
        if isinstance(output, dict):
            for msg in output.get("messages", []):
                if isinstance(msg, dict) and isinstance(msg.get("body"), str):
                    msg["body"] = msg["body"][:_COMPOSER_BODY_LIMIT]
        compacted.append(data)
    return compacted


def build_composer_messages(plan: Any, tool_results: list[Any], preferences: list[Any],
                            user_request: str, tz_name: str) -> list[BaseMessage]:
    """Build response-composer prompt with plan, results, user prefs, request, and timezone."""
    plan_text = plan.model_dump_json() if hasattr(plan, "model_dump_json") else json.dumps(plan)
    results_text = json.dumps(_compact_results(tool_results))
    prefs_text = "\n".join(str(p) for p in preferences[:3]) if preferences else "No preferences stored."
    sys = f"{_COMPOSER_SYS}\n<user_style>\n{prefs_text}\n</user_style>"
    human = (f"<context>\nTimezone: {tz_name}\n</context>\n\n"
             f"<plan>\n{plan_text}\n</plan>\n\n<results>\n{results_text}\n</results>\n\n"
             f"<request>\n{user_request}\n</request>")
    return [SystemMessage(content=sys), HumanMessage(content=human)]


def build_direct_answer_messages(user_request: str, tool_specs: list[Any], preferences: list[Any]) -> list[BaseMessage]:
    """Build prompt for direct conversational answers when no tools are needed."""
    tools_block = json.dumps([t.model_dump() if hasattr(t, "model_dump") else t for t in tool_specs])
    prefs_text = "\n".join(str(p) for p in preferences[:3]) if preferences else "No preferences stored."
    sys = f"{_DIRECT_ANSWER_SYS}\n<user_style>\n{prefs_text}\n</user_style>"
    human = f"<tools>\n{tools_block}\n</tools>\n\n<request>\n{user_request}\n</request>"
    return [SystemMessage(content=sys), HumanMessage(content=human)]
