"""Conversation-memory prompt wiring tests (REVIEW.md R12).

Covers the pure history rendering into the planner prompt. The repository
round-trip (save_message/get_recent_messages) needs Postgres and is left to
integration testing, like the other repositories.
"""

from __future__ import annotations

from app.prompts import _history_block, build_planner_messages

_NOW = "2026-06-14T09:00:00+05:30"
_HISTORY = [
    {"role": "user", "content": "schedule a sync with Priya tomorrow 3pm"},
    {"role": "assistant", "content": "Booked the sync with Priya."},
]


def _planner_human(history):
    return build_planner_messages(
        user_request="move that meeting to 4pm", examples=[], tool_specs=[],
        now_local=_NOW, tz_name="Asia/Kolkata", weekday_ref="ref",
        history=history)[-1].content


def test_history_block_empty_when_no_history():
    assert _history_block([]) == ""


def test_history_block_renders_roles_and_content():
    block = _history_block(_HISTORY)
    assert block.startswith("<history>")
    assert "user: schedule a sync with Priya" in block
    assert "assistant: Booked the sync with Priya." in block


def test_history_block_truncates_long_content():
    long = [{"role": "user", "content": "x" * 500}]
    assert "x" * 200 in _history_block(long)
    assert "x" * 201 not in _history_block(long)


def test_planner_prompt_includes_history_when_present():
    assert "<history>" in _planner_human(_HISTORY)


def test_planner_prompt_omits_history_when_absent():
    human = _planner_human([])
    assert "<history>" not in human
    assert "<request>" in human  # rest of the prompt intact
