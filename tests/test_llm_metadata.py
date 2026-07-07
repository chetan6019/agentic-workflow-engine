"""LLM client metadata tests: Langfuse grouping body + cached-client isolation.

get_llm with run metadata must attach LiteLLM's Langfuse grouping fields
(trace_id, session_id, trace_user_id, generation_name) plus the body-level
`user` for spend attribution — without touching the cached base client.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.state import AgentState
from app.llm import client as llm_client
from app.prompts import PROMPT_VERSION


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    """Stub settings and isolate the lru_cache so tests never need a .env."""
    monkeypatch.setattr(
        llm_client, "get_settings",
        lambda: SimpleNamespace(litellm_url="http://litellm:4000",
                                litellm_virtual_key="sk-test"),
    )
    llm_client._base_llm.cache_clear()
    yield
    llm_client._base_llm.cache_clear()


def test_metadata_builds_langfuse_grouping_body():
    llm = llm_client.get_llm(
        "composer", {"trace_id": "t1", "session_id": "s1", "user_id": "u1"})
    assert llm.extra_body == {
        "metadata": {"generation_name": "composer",
                     "tags": ["role:composer", f"prompt_version:{PROMPT_VERSION}"],
                     "prompt_version": PROMPT_VERSION,
                     "trace_id": "t1", "session_id": "s1", "trace_user_id": "u1"},
        "user": "u1",
    }


def test_without_metadata_returns_cached_client_with_no_extra_body():
    first = llm_client.get_llm("composer")
    second = llm_client.get_llm("composer")
    assert first is second  # lru_cache hit
    assert first.extra_body is None


def test_metadata_copy_never_mutates_the_cached_client():
    base = llm_client.get_llm("composer")
    llm_client.get_llm("composer", {"trace_id": "t", "session_id": "s", "user_id": "u"})
    assert base.extra_body is None


def test_run_metadata_extracts_the_state_fields():
    state = AgentState(trace_id="t", user_id="u", session_id="sess", user_request="hi")
    assert llm_client.run_metadata(state) == {
        "trace_id": "t", "session_id": "sess", "user_id": "u"}
