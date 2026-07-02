"""Tests for prompt-size compaction: composer/guard tool-output trimming + planner slimming.

Pure functions, no infra. These guard the token-budget reductions that keep the composer
(shared groq/gpt-oss-120b) under Groq's per-minute token limit.
"""

from __future__ import annotations

import json

from app import prompts


def _result(output: dict) -> dict:
    return {"step_id": "s1", "ok": True, "output": output, "error": None, "latency_ms": 5}


def test_compact_clips_long_strings():
    # Email bodies are the most common long string; they clip to _RESULT_STR_LIMIT (+ "…").
    out = prompts._compact_results([_result({"messages": [{"body": "x" * 1000}]})])
    body = out[0]["output"]["messages"][0]["body"]
    assert len(body) == prompts._RESULT_STR_LIMIT + 1 and body.endswith("…")


def test_compact_caps_long_lists():
    out = prompts._compact_results([_result({"items": [{"id": i} for i in range(20)]})])
    items = out[0]["output"]["items"]
    assert len(items) == prompts._RESULT_LIST_LIMIT + 1  # 5 kept + a "…+N more" marker
    assert isinstance(items[-1], str) and items[-1].startswith("…+")


def test_compact_caps_big_dict_keys():
    big = {"metric": {f"k{i}": "v" for i in range(500)}}  # large finnhub-style payload
    capped = prompts._compact_results([_result(big)])[0]["output"]
    metric = capped["metric"]
    assert metric["_more_fields"] == 500 - prompts._RESULT_MAX_KEYS
    # Serialized output is bounded, not the original 500-field blob.
    assert len(json.dumps(capped)) < len(json.dumps(big))


def test_compact_passes_small_output_through():
    small = {"quote": {"c": 150.0}}
    assert prompts._compact_results([_result(small)])[0]["output"] == small


def test_slim_tool_specs_clips_description():
    specs = [{"name": "t", "description": "d" * 500, "server": "google", "input_schema": {}}]
    slim = prompts._slim_tool_specs(specs)
    assert len(slim[0]["description"]) == prompts._TOOL_DESC_LIMIT
    assert slim[0]["server"] == "google"


def test_compact_keeps_request_relevant_items():
    # Contextual compression: with a user request, an oversized list keeps the
    # matching item ("priya" at position 8) instead of blindly the first 5.
    emails = [{"from": f"person{i}@x.com", "subject": "weekly digest"} for i in range(8)]
    emails.append({"from": "priya@x.com", "subject": "budget reply"})
    out = prompts._compact_results(
        [_result({"messages": emails})], user_request="did priya reply about the budget?")
    kept = out[0]["output"]["messages"]
    assert len(kept) == prompts._RESULT_LIST_LIMIT + 1  # limit kept + "…+N more" marker
    assert any(isinstance(m, dict) and m.get("from") == "priya@x.com" for m in kept)


def test_compact_without_request_keeps_first_n():
    # Back-compat: no user request → original first-N behavior, original order.
    items = [{"id": i} for i in range(20)]
    out = prompts._compact_results([_result({"items": items})])
    kept = out[0]["output"]["items"]
    assert kept[: prompts._RESULT_LIST_LIMIT] == items[: prompts._RESULT_LIST_LIMIT]


def test_guard_and_composer_see_identical_evidence():
    # The guard must judge against exactly the evidence the composer saw.
    emails = [{"from": f"p{i}@x.com", "body": "hello " * 100} for i in range(9)]
    results = [_result({"messages": emails})]
    request = "summarize the email from p7"
    composer_msgs = prompts.build_composer_messages(
        plan={"steps": []}, tool_results=results, preferences=[],
        user_request=request, tz_name="Asia/Kolkata")
    guard_msgs = prompts.build_guard_judge_messages(
        draft={"summary": "s"}, user_request=request, tool_results=results)
    composer_results = composer_msgs[1].content.split("<results>")[1].split("</results>")[0]
    guard_evidence = guard_msgs[1].content.split("<evidence>")[1].split("</evidence>")[0]
    assert composer_results.strip() == guard_evidence.strip()


def test_direct_answer_tools_are_slimmed():
    # Full JSON Schemas must not reach the direct-answer prompt (token waste).
    specs = [{"name": "send_email", "description": "d", "server": "google",
              "input_schema": {"properties": {"to": {"type": "string",
                                                     "description": "recipient " * 50}},
                               "required": ["to"]}}]
    msgs = prompts.build_direct_answer_messages(
        user_request="what can you do?", tool_specs=specs, preferences=[])
    human = msgs[1].content
    assert '"params": ["to"]' in human          # slim shape present
    assert "recipient recipient" not in human   # schema description text absent


def test_planner_prompt_ends_with_volatile_blocks():
    # Prefix caching: stable blocks (tools/examples) must precede volatile ones
    # (weekday table, current time, request) in the human message.
    msgs = prompts.build_planner_messages(
        user_request="hi", examples=[], tool_specs=[],
        now_local="2026-07-02T10:00:00+05:30", tz_name="Asia/Kolkata",
        weekday_ref="Monday = 2026-07-06")
    human = msgs[1].content
    assert human.index("<tools>") < human.index("<examples>")
    assert human.index("<examples>") < human.index("<weekday_reference>")
    assert human.index("<weekday_reference>") < human.index("<context>")
    assert human.index("<context>") < human.index("<request>")
