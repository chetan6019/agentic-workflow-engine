"""Pure prompt builder functions — no DB, LLM, or HTTP imports.

Layout of this file (top to bottom):
  1. Versioning + fingerprint (trace observability).
  2. Size limits shared by the compaction helpers.
  3. Shared helpers used by more than one agent.
  4. One section per agent — PLANNER, RETRIEVER, GUARD, COMPOSER, DIRECT ANSWER —
     each holding its system prompt, its private helpers, and its build_* function.

The external tools the planner can use are the four MCP servers:
  google (Gmail + Calendar), github, reddit, finnhub (stocks, read-only).
Whenever a tool/server is added or renamed, update BOTH _PLANNER_SYS and the
friendly names in _RETRIEVER_ROUTER_SYS / _DIRECT_ANSWER_SYS, then bump
PROMPT_VERSION.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# ---------------------------------------------------------------------------
# 1. Versioning + fingerprint
# ---------------------------------------------------------------------------

# Version-locked prompt set. Bump on every change to any *_SYS string, slim/compact
# helper that affects rendered output, or schema referenced from a prompt. Langfuse
# traces and eval runs attach this so a regression can be attributed to an exact
# prompt revision (not just a git SHA, which also moves on unrelated edits).
PROMPT_VERSION = "2026-07-02.2"

PROMPT_CHANGELOG = """\
2026-07-02.2  Token-reduction pass. Planner human message reordered stable-first
              (tools, examples before history, weekday table, current time,
              request) so providers with automatic prefix caching (OpenAI,
              Gemini) get cache hits — the provider-generic lever; no
              provider-specific cache markers in app code. Direct-answer prompt
              now uses slim tool specs instead of full JSON Schemas. History
              block gained rolling compaction (recent turns verbatim, older
              turns tightly clipped). Tool-output compaction is now
              request-aware: an oversized list keeps the items most relevant to
              the user request instead of the first N; composer and guard
              receive the SAME compacted evidence.
2026-07-02.1  Retriever-router + direct-answer prompts: stale Notion/Slack tool
              names replaced with the live google/github/reddit/finnhub set.
              Planner prompt restructured into labeled rule groups (TOOL AND
              ACTION RULES / DATES AND TIMES / IDENTIFIERS / ...) — same rules,
              clearer layout. File reorganised into per-agent sections.
2026-06-24.1  Guard prompt: abstain-at-0.5 convention + NEVER list.
              Composer prompt: structured ok=false failure-mode rules + NEVER list.
              Retrieval examples: tool-signature dedup so near-duplicate past plans
              don't both reach the planner.
              Version + fingerprint helpers exposed for trace observability.
2026-06-22.0  Planner tool enum migrated to google/github/reddit/finnhub
              (R13 stale block kept inside _PLANNER_SYS).
"""


def fingerprint(messages: list[BaseMessage]) -> str:
    """Short, stable hash of a rendered prompt — emitted alongside PROMPT_VERSION
    on every LLM call so identical prompts group together in traces regardless of
    surrounding metadata. Use the first 12 hex chars; collisions are not a concern
    at this scale and shorter IDs read better in logs.
    """
    raw = "".join(str(m.content) for m in messages).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 2. Size limits
# ---------------------------------------------------------------------------

# Retrieved past-plan text (request + summary) can come from OTHER users after the
# retriever broadens its search, so it is untrusted input. Clip its length and
# neutralise prompt-tag characters before it nears the planner instructions (R33).
_RETRIEVED_TEXT_LIMIT = 300
# Tool-output compaction (composer + guard). _shrink bounds every output: clip long string
# values, cap list and dict sizes, and limit nesting depth — so one big result (finnhub
# financials, a long email/calendar list) can't blow the model's token limit. The Gmail email
# body is just the most common long string this catches. See _compact_results.
_RESULT_STR_LIMIT = 200
_RESULT_LIST_LIMIT = 5
_RESULT_MAX_KEYS = 40
_RESULT_MAX_DEPTH = 6
# Verbose tool docstrings bloat the planner's tool listing; clip each description.
_TOOL_DESC_LIMIT = 200
# Rolling history compaction: the newest turns are rendered verbatim (clipped to
# _HISTORY_TURN_LIMIT chars each); anything older is compacted to a tight
# _HISTORY_OLD_TURN_LIMIT-char line inside <history_summary>. This lets the API
# hand the planner a wider window without the prompt growing linearly.
_HISTORY_RECENT_TURNS = 4
_HISTORY_TURN_LIMIT = 200
_HISTORY_OLD_TURN_LIMIT = 80
# Request-aware list compaction: words shorter than this never count toward
# relevance overlap ("a", "to", "of" would match everything).
_RELEVANCE_MIN_WORD_LEN = 3

# ---------------------------------------------------------------------------
# 3. Shared helpers (used by more than one agent's builder)
# ---------------------------------------------------------------------------

def _query_words(user_request: str | None) -> frozenset[str]:
    """Lowercased content words of the request, used for relevance overlap scoring."""
    if not user_request:
        return frozenset()
    words = "".join(c if c.isalnum() else " " for c in user_request.lower()).split()
    return frozenset(w for w in words if len(w) >= _RELEVANCE_MIN_WORD_LEN)

def _relevance(item: Any, query_words: frozenset[str]) -> int:
    """How many request words appear in the item's JSON text. Cheap, no LLM."""
    try:
        text = json.dumps(item).lower()
    except (TypeError, ValueError):
        text = str(item).lower()
    return sum(1 for w in query_words if w in text)

def _pick_relevant(items: list[Any], query_words: frozenset[str]) -> list[Any]:
    """Choose _RESULT_LIST_LIMIT items from an oversized list, keeping original order.

    With query words available this is contextual compression: keep the items that
    overlap the user's request most (so "did Priya reply?" keeps the Priya email even
    if it's 8th in the list). Without them, fall back to the first N as before.
    """
    if not query_words:
        return items[:_RESULT_LIST_LIMIT]
    scored = [(idx, _relevance(item, query_words)) for idx, item in enumerate(items)]
    # Highest overlap wins; ties break toward earlier position (stable, deterministic).
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    kept_indices = sorted(idx for idx, _score in scored[:_RESULT_LIST_LIMIT])
    return [items[idx] for idx in kept_indices]

def _shrink(value: Any, depth: int = 0, query_words: frozenset[str] = frozenset()) -> Any:
    """Recursively bound a tool-output value so its serialized JSON stays small.

    Clips long strings, caps list and dict sizes, and limits nesting depth. The structure is
    preserved (the composer/guard can still read fields), but the output can't explode — this
    is the catch-all for big payloads like finnhub financials or a long email/calendar list.
    When query_words is non-empty, oversized lists keep the most request-relevant items
    (contextual compression) instead of blindly the first N.
    """
    if depth >= _RESULT_MAX_DEPTH:
        return "…"
    if isinstance(value, str):
        return value if len(value) <= _RESULT_STR_LIMIT else value[:_RESULT_STR_LIMIT] + "…"
    if isinstance(value, list):
        kept = value if len(value) <= _RESULT_LIST_LIMIT else _pick_relevant(value, query_words)
        shrunk = [_shrink(v, depth + 1, query_words) for v in kept]
        if len(value) > _RESULT_LIST_LIMIT:
            shrunk.append(f"…+{len(value) - _RESULT_LIST_LIMIT} more")
        return shrunk
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for i, (key, val) in enumerate(value.items()):
            if i >= _RESULT_MAX_KEYS:
                out["_more_fields"] = len(value) - _RESULT_MAX_KEYS
                break
            out[str(key)] = _shrink(val, depth + 1, query_words)
        return out
    return value

def _compact_results(tool_results: list[Any], user_request: str | None = None) -> list[Any]:
    """Shrink tool outputs so the composer/guard prompt stays within the model's token limit.

    Bounds every tool output via _shrink (clip strings, cap list/dict sizes). When
    user_request is given, oversized lists keep the most request-relevant items rather
    than the first N. Applied to BOTH the composer and the guard-judge prompts with the
    SAME user_request — the guard must judge against exactly the evidence the composer
    saw, or grounded facts would look hallucinated.
    """
    query_words = _query_words(user_request)
    compacted: list[Any] = []
    for r in tool_results:
        data = r.model_dump() if hasattr(r, "model_dump") else dict(r)
        output = data.get("output") if isinstance(data, dict) else None
        if isinstance(output, dict):
            data["output"] = _shrink(output, query_words=query_words)
        compacted.append(data)
    return compacted

def _revision_block(judge_feedback: str | None) -> str:
    """Optional correction note appended to a re-composition prompt (else empty)."""
    return f"\n\n<revision_note>\n{judge_feedback}\n</revision_note>" if judge_feedback else ""

def _prefs_text(preferences: list[Any]) -> str:
    """Render up to 3 stored user-style preferences for the composer/direct-answer prompts."""
    if not preferences:
        return "No preferences stored."
    return "\n".join(str(p) for p in preferences[:3])

# ---------------------------------------------------------------------------
# 4. PLANNER — turns a user request into a typed ExecutionPlan of tool calls
# ---------------------------------------------------------------------------

_PLANNER_SYS = """<instructions>
You are a workflow planner. Given a user request, retrieved similar plans, and available tools,
produce a typed ExecutionPlan. Use the minimum steps needed. Be explicit about dependencies.
Prefer parallel execution when steps are independent.

TOOL AND ACTION RULES:
- For EVERY step: `tool` MUST be the chosen tool's `server` value (one of: google, github,
  reddit, finnhub) and `action` MUST be that tool's `name` (e.g. send_email, search_email).
  NEVER put the action name in `tool`.
- Put the tool's parameters in `arguments` (omit user_id — it is injected automatically).
- When the user asks for a specific NUMBER of items ("top 3 unread emails", "my latest
  5 messages", "3 events"), set the tool's result-count argument (e.g. `max_results`)
  to that exact number. When no count is requested, omit it and let the tool default.

DATES AND TIMES:
- When a step needs a date or time (e.g. google create_event start/end), interpret every
  relative or clock time ("tomorrow 9am", "next Monday") relative to the current moment and
  timezone given in <context>, and emit it as an RFC3339 timestamp that INCLUDES that
  timezone's UTC offset (e.g. 2026-06-11T09:00:00+05:30). Never emit a bare-UTC "Z" time
  for a local clock time.
- For any named weekday ("Monday", "coming Monday", "this Friday", "next Tuesday") you MUST
  use the EXACT date listed in <weekday_reference> — never compute the weekday yourself.
  "Coming"/"next"/"this <weekday>" all map to that weekday's date in the reference.

IDENTIFIERS:
- NEVER invent identifiers you were not explicitly given (event_id, message_id, page_id,
  etc.). A made-up id will fail.
- To update or delete an EXISTING calendar event the user only describes ("the 1:1 with
  Priya tomorrow"), OMIT event_id entirely and instead pass `match_summary` (distinctive
  words from the event title, e.g. "Priya") plus `time_min` and `time_max` RFC3339
  timestamps bounding the day it falls on; the tool resolves the real id.

CONVERSATION HISTORY:
- If <history> is present, use it to resolve references to earlier turns
  ("that meeting", "reply to it", "the same people") — but never invent ids from it.

NO-TOOL REQUESTS:
- If the request needs NO tools — a greeting, small talk, a thank-you, or a question you can
  answer directly from your own knowledge or <history> — return an EMPTY `steps` list with
  `complexity_score` 0 and `strategy` "sequential". A later step composes the direct reply.

CONFIDENCE:
- Rate EVERY step with "confidence": "high" | "medium" | "low".
  high = a routine read, or a write whose arguments the user stated explicitly;
  medium = arguments are ambiguous or partly inferred from context/history;
  low = you are guessing at what the user wants.
  Medium/low-confidence writes PAUSE for human approval before running, so rate
  honestly — never inflate to high to skip the pause.
</instructions>
<output_format>
Output ONLY valid json, no prose, matching exactly:
{"reasoning": "string",
 "steps": [{"id": "string", "tool": "google", "action": "send_email",
            "arguments": {}, "depends_on": ["string"], "confidence": "high|medium|low"}],
 "strategy": "sequential|parallel|mixed",
 "complexity_score": 5, "estimated_cost_usd": 0.0, "requires_approval": false}
Example — send an email (recipient and text given explicitly):
{"id":"s1","tool":"google","action":"send_email",
 "arguments":{"to":"a@b.com","subject":"Hi","body":"Hello"},"depends_on":[],
 "confidence":"high"}
Example — "reply to that thread" (recipient inferred from history):
{"id":"s1","tool":"google","action":"reply_thread",
 "arguments":{"thread_id":"t123","body":"Sounds good."},"depends_on":[],
 "confidence":"medium"}
Example — greeting / no tools needed:
{"reasoning":"User only greeted; no tools needed.","steps":[],
 "strategy":"sequential","complexity_score":0,"estimated_cost_usd":0.0,"requires_approval":false}
</output_format>"""

def _sanitize_retrieved(text: Any) -> str:
    """Clip length and escape angle brackets in untrusted retrieved text (R33)."""
    s = str(text or "")
    if len(s) > _RETRIEVED_TEXT_LIMIT:
        s = s[:_RETRIEVED_TEXT_LIMIT] + "…"
    return s.replace("<", "‹").replace(">", "›")

def _slim_tool_specs(tool_specs: list[Any]) -> list[dict[str, Any]]:
    """Reduce tool specs to name/description/server + param names; full JSON Schemas bloat the prompt."""
    slim: list[dict[str, Any]] = []
    for t in tool_specs:
        d = t.model_dump() if hasattr(t, "model_dump") else dict(t)
        schema = d.get("input_schema") or {}
        props = list(schema.get("properties", {})) if isinstance(schema, dict) else []
        required = schema.get("required", []) if isinstance(schema, dict) else []
        slim.append({"name": d.get("name"),
                     "description": str(d.get("description") or "")[:_TOOL_DESC_LIMIT],
                     "server": d.get("server"), "params": props, "required": required})
    return slim

def _signature(plan: Any) -> str:
    """Compact (tool.action|tool.action|…) signature of a past plan.

    Used only to deduplicate retrieved examples — two past plans that hit the same
    sequence of tool actions teach the planner nothing extra and waste tokens.
    Robust to plans stored as either dict or BaseModel.
    """
    if plan is None:
        return ""
    obj = plan.model_dump() if hasattr(plan, "model_dump") else plan
    if not isinstance(obj, dict):
        return ""
    steps = obj.get("steps") or []
    parts: list[str] = []
    for s in steps:
        if isinstance(s, dict):
            parts.append(f"{s.get('tool')}.{s.get('action')}")
    return "|".join(parts)

def _slim_examples(examples: list[Any]) -> list[dict[str, Any]]:
    """Keep up to 2 retrieved plans, deduped by tool-call signature, sanitized (R33).

    Earlier draft just took the first 2 with no dedup; when the retriever returned
    two near-identical past plans (common after broadening) the planner saw the same
    pattern twice and learned nothing new. Now we walk the list in order and skip
    any plan whose (tool.action|…) signature we've already kept.
    """
    out: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for e in examples:
        d = e.model_dump() if hasattr(e, "model_dump") else dict(e)
        sig = _signature(d.get("plan_json"))
        # Empty signature = no usable plan steps; keep it anyway so a no-tools
        # example (e.g. greetings) can still teach the planner that pattern,
        # but only once per call.
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        out.append({"request": _sanitize_retrieved(d.get("request_text")),
                    "summary": _sanitize_retrieved(d.get("summary"))})
        if len(out) == 2:
            break
    return out

def _history_block(history: list[dict[str, str]]) -> str:
    """Render recent turns so follow-ups ('move that meeting') resolve; empty when none.

    Rolling compaction: the newest _HISTORY_RECENT_TURNS turns are rendered verbatim
    (clipped to _HISTORY_TURN_LIMIT chars); older turns are compacted to
    _HISTORY_OLD_TURN_LIMIT-char lines inside <history_summary>. Callers can therefore
    pass a wider window without the planner prompt growing linearly with chat length.
    """
    if not history:
        return ""
    older = history[:-_HISTORY_RECENT_TURNS]
    recent = history[-_HISTORY_RECENT_TURNS:]
    block = ""
    if older:
        old_lines = [f"{t.get('role', '?')}: {str(t.get('content', ''))[:_HISTORY_OLD_TURN_LIMIT]}"
                     for t in older]
        block += "<history_summary>\n" + "\n".join(old_lines) + "\n</history_summary>\n\n"
    lines = [f"{t.get('role', '?')}: {str(t.get('content', ''))[:_HISTORY_TURN_LIMIT]}"
             for t in recent]
    return block + "<history>\n" + "\n".join(lines) + "\n</history>\n\n"

def build_planner_messages(user_request: str, examples: list[Any], tool_specs: list[Any],
                           now_local: str, tz_name: str, weekday_ref: str,
                           history: list[dict[str, str]] | None = None) -> list[BaseMessage]:
    """Build planner prompt with tool listing, retrieved-plan examples, and time context.

    Block order is stable-first for prefix caching: the system prompt is byte-identical
    across calls, then tools/examples (repeat for similar requests), then history, and
    only at the END the parts that change every call (weekday table, current time,
    request). Providers with automatic prefix caching (OpenAI, Gemini) bill the shared
    prefix at the cached rate; keep new blocks in most-stable-first position.
    """
    tools_block = json.dumps(_slim_tool_specs(tool_specs))
    examples_block = json.dumps(_slim_examples(examples))
    context = f"Current time: {now_local}\nTimezone: {tz_name}"
    human = (f"<tools>\n{tools_block}\n</tools>\n\n"
             f"<examples>\n{examples_block}\n</examples>\n\n"
             f"{_history_block(history or [])}"
             f"<weekday_reference>\n{weekday_ref}\n</weekday_reference>\n\n"
             f"<context>\n{context}\n</context>\n\n"
             f"<request>\n{user_request}\n</request>")
    return [SystemMessage(content=_PLANNER_SYS), HumanMessage(content=human)]

# ---------------------------------------------------------------------------
# 5. RETRIEVER — routes/rewrites the search query, then grades candidates
# ---------------------------------------------------------------------------

_RETRIEVER_ROUTER_SYS = """<instructions>
You are the retrieval router for a workflow planner.
In ONE pass: decide whether searching past workflow plans would help fulfil the
request, AND rewrite the request into a concise, keyword-rich search query for
semantic retrieval. Retrieval helps for task-like requests (email, calendar,
GitHub, Reddit, or stock-data actions); it does not help for greetings, meta
questions, or small talk.
Output JSON: {"should_retrieve": bool, "query": str}.
`query` must always be filled, even when should_retrieve is false.
</instructions>"""

_RETRIEVER_GRADER_SYS = """<instructions>
You are a relevance grader for workflow plan retrieval.
Score each candidate for relevance to the query, 0.0 (unrelated) to 1.0 (same task).
Output JSON: {"hits": [{"plan_id": str, "relevance": float}]}.
</instructions>"""

def build_retriever_router_messages(user_request: str) -> list[BaseMessage]:
    """Build the combined should-retrieve + query-rewrite prompt (one LLM call)."""
    return [SystemMessage(content=_RETRIEVER_ROUTER_SYS),
            HumanMessage(content=f"<request>\n{user_request}\n</request>")]

def build_retriever_grader_messages(query: str, candidates: list[Any]) -> list[BaseMessage]:
    """Build grader prompt scoring each candidate's relevance to the query."""
    cands = json.dumps([c.model_dump() if hasattr(c, "model_dump") else c for c in candidates])
    human = f"<request>\n{query}\n</request>\n<candidates>\n{cands}\n</candidates>"
    return [SystemMessage(content=_RETRIEVER_GRADER_SYS), HumanMessage(content=human)]

# ---------------------------------------------------------------------------
# 6. GUARD — judges the draft response against the tool-result evidence
# ---------------------------------------------------------------------------

_GUARD_SYS = """<instructions>
You are a quality judge for AI workflow responses. You are given the user request, the draft response, and <evidence> — the raw tool
results the draft is supposed to be based on. Score three values in [0.0, 1.0]:
- tone_fit: alignment with the user's request and style.
- hallucination_risk: the degree to which the draft makes claims that are INVENTED or
  CONTRADICT the <evidence>. Summarizing, paraphrasing, condensing, or rewording the
  evidence is EXPECTED and is NOT hallucination — only the underlying facts need to hold.
  Raise risk only when a concrete fact — a name, email, time, count, status, or id — is
  ABSENT from or CONTRADICTS the evidence (e.g. an email/event that isn't there, a wrong
  recipient, an invented number). A faithful confirmation of a SUCCESSFUL action ("Sent
  your email to …", "Created the event") is grounded when the evidence is that action's
  success receipt — score it low. High risk = fabricated or contradicted facts; low risk =
  every fact is supported even if the wording differs. If <evidence> is empty (a direct
  conversational answer with no tool calls), judge plausibility instead.
- instruction_adherence: how fully the draft satisfies the request.

ABSTAIN RULE — when you cannot decide a score with reasonable confidence (evidence is
ambiguous, partial, the draft is unusual, or you would otherwise be guessing), set ALL
THREE scores to exactly 0.5. The orchestrator treats 0.5 as "needs human review". DO
NOT guess high (which would auto-approve) or low (which would reject) on uncertainty.

NEVER:
- Penalize correct paraphrasing or summarization as hallucination — only fabricated or
  contradicted FACTS raise hallucination_risk.
- Penalize a politely declined or "I cannot help with that" answer as low
  instruction_adherence when declining was the correct response.
- Reward draft length; brevity is fine if the evidence supports it.
- Score a successful-action confirmation ("Sent your email") as hallucinated when the
  evidence is that action's success receipt.

Output JSON: {"tone_fit": float, "hallucination_risk": float, "instruction_adherence": float}.
</instructions>"""

def build_guard_judge_messages(draft: Any, user_request: str,
                               tool_results: list[Any] | None = None) -> list[BaseMessage]:
    """Build guard-judge prompt: draft + original request + the tool-result evidence.

    The evidence lets the judge ground hallucination_risk in what the tools
    actually returned (compacted to trim long email bodies) rather than guessing
    from the draft alone. Empty for direct/meta answers that ran no tools.
    """
    draft_text = draft.model_dump_json() if hasattr(draft, "model_dump_json") else json.dumps(draft)
    evidence = json.dumps(_compact_results(tool_results or [], user_request))
    human = (f"<draft>\n{draft_text}\n</draft>\n\n"
             f"<evidence>\n{evidence}\n</evidence>\n\n"
             f"<request>\n{user_request}\n</request>")
    return [SystemMessage(content=_GUARD_SYS), HumanMessage(content=human)]

# ---------------------------------------------------------------------------
# 7. COMPOSER — writes the user-facing answer from the plan + tool results
# ---------------------------------------------------------------------------

_COMPOSER_SYS = """<instructions>
You are a workflow response composer. Summarise what was done clearly and honestly.
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

FAILURE MODE — when a tool result has ok=false:
- Acknowledge the failure plainly in `detail_markdown` (e.g. "I couldn't send the
  email — Gmail returned: <human-language explanation of the error>"). Translate
  technical error codes into user-friendly language; never paste raw stack traces
  or HTTP status codes.
- Put the failed action in `actions_pending`, NOT `actions_taken`. `actions_taken`
  is reserved for actions that completed successfully.
- Suggest ONE concrete next step ("Re-connect Gmail under Settings", "Check that
  the recipient address is valid", "Try again in a few minutes — the upstream is
  rate-limiting"). Never a generic "please try again later".
- If multiple steps failed, list each in `actions_pending` with its own
  one-line explanation in `detail_markdown`.

NEVER:
- Fabricate a success message — every claim in `detail_markdown` must be supported
  by a tool result with ok=true (or be a direct conversational answer with no tools).
- Hide that a step failed by omitting it from the response.
- Use technical error codes verbatim ("HTTP 429", "RESOURCE_EXHAUSTED") — translate
  to user language ("Gmail is rate-limiting us right now").
- Leave `detail_markdown` empty, even on total failure.
- Mention internal tool names, step ids, or the orchestrator to the user.
</instructions>
<output_format>
Output ONLY valid json, no prose, matching exactly:
{"summary": "string", "detail_markdown": "string", "actions_taken": ["string"],
 "actions_pending": ["string"], "citations": ["string"]}
`citations` may ONLY contain http(s) URLs that literally appear in the tool
results. It is almost always an empty list. NEVER put step ids ("s1"), tool
names, or any internal identifier in `citations`.
</output_format>"""

def build_composer_messages(plan: Any, tool_results: list[Any], preferences: list[Any],
                            user_request: str, tz_name: str,
                            judge_feedback: str | None = None) -> list[BaseMessage]:
    """Build response-composer prompt with plan, results, user prefs, request, and timezone."""
    plan_text = plan.model_dump_json() if hasattr(plan, "model_dump_json") else json.dumps(plan)
    results_text = json.dumps(_compact_results(tool_results, user_request))
    sys = f"{_COMPOSER_SYS}\n<user_style>\n{_prefs_text(preferences)}\n</user_style>"
    human = (f"<context>\nTimezone: {tz_name}\n</context>\n\n"
             f"<plan>\n{plan_text}\n</plan>\n\n<results>\n{results_text}\n</results>\n\n"
             f"<request>\n{user_request}\n</request>{_revision_block(judge_feedback)}")
    return [SystemMessage(content=sys), HumanMessage(content=human)]

# ---------------------------------------------------------------------------
# 8. DIRECT ANSWER — conversational/meta replies when the plan needs no tools
# ---------------------------------------------------------------------------

_DIRECT_ANSWER_SYS = """<instructions>
You are the assistant for a workflow automation tool. The user has asked a
conversational or meta question that does not require running any tools —
answer it directly in plain language.

Use the tool catalog below as ground truth for what you can and cannot do. Be
honest: never claim a capability that isn't represented in the catalog.
Reference integrations by friendly name (Gmail, Google Calendar, GitHub,
Reddit, Finnhub) rather than internal tool names.
</instructions>
<output_format>
Output ONLY valid json, no prose, matching exactly:
{"summary": "string", "detail_markdown": "string", "actions_taken": [],
 "actions_pending": [], "citations": []}
`summary` is a one-line answer. `detail_markdown` is the full friendly answer
shown to the user (Markdown allowed). `actions_taken`, `actions_pending`,
`citations` must be empty lists.
</output_format>"""

def build_direct_answer_messages(user_request: str, tool_specs: list[Any], preferences: list[Any],
                                 judge_feedback: str | None = None) -> list[BaseMessage]:
    """Build prompt for direct conversational answers when no tools are needed.

    Uses the same slim tool specs as the planner — the model only needs names,
    descriptions, and param names to describe its capabilities; full JSON Schemas
    (12 specs here) were the single largest token waste in this prompt.
    """
    tools_block = json.dumps(_slim_tool_specs(tool_specs))
    sys = f"{_DIRECT_ANSWER_SYS}\n<user_style>\n{_prefs_text(preferences)}\n</user_style>"
    human = (f"<tools>\n{tools_block}\n</tools>\n\n"
             f"<request>\n{user_request}\n</request>{_revision_block(judge_feedback)}")
    return [SystemMessage(content=sys), HumanMessage(content=human)]
