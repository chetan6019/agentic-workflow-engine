"""Unit tests for the GitHub MCP server — PyGithub fully faked, no real API, no network.

We patch ``_client`` to return a hand-built fake Github object and ``resolve_user_token`` to a
canned token, so the real PyGithub client is never constructed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.mcp.servers import github_server as gh


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    async def _token(user_id: str, provider: str) -> str:
        assert provider == "github"
        return "tok"

    monkeypatch.setattr(gh, "resolve_user_token", _token)


def _patch_client(monkeypatch, fake: object) -> None:
    monkeypatch.setattr(gh, "_client", lambda token: fake)


class _FakeRepo:
    def __init__(self) -> None:
        self.created: list[dict] = []

    def get_pull(self, number: int):
        return SimpleNamespace(number=number, title="Fix bug", state="open",
                               body="details", html_url="http://pr", user=SimpleNamespace(login="me"))

    def get_issue(self, number: int):
        return SimpleNamespace(
            create_comment=lambda body: SimpleNamespace(id=99, html_url="http://comment"))

    def create_issue(self, title: str, body: str):
        self.created.append({"title": title, "body": body})
        return SimpleNamespace(number=7, html_url="http://issue")

    def get_commits(self):
        return [SimpleNamespace(sha="abc", html_url="http://commit",
                                commit=SimpleNamespace(message="msg",
                                                       author=SimpleNamespace(name="me")))]


async def test_list_my_prs(monkeypatch):
    pr = SimpleNamespace(repository=SimpleNamespace(full_name="o/r"), number=3,
                         title="t", html_url="http://pr")
    fake = SimpleNamespace(get_user=lambda: SimpleNamespace(login="me"),
                           search_issues=lambda q: [pr])
    _patch_client(monkeypatch, fake)
    result = await gh.list_my_prs("u", state="open")
    assert result["pull_requests"][0]["repo"] == "o/r"
    assert result["pull_requests"][0]["number"] == 3


async def test_get_pr(monkeypatch):
    fake = SimpleNamespace(get_repo=lambda repo: _FakeRepo())
    _patch_client(monkeypatch, fake)
    result = await gh.get_pr("u", "o/r", 12)
    assert result["number"] == 12 and result["author"] == "me" and result["state"] == "open"


async def test_comment_on_pr(monkeypatch):
    fake = SimpleNamespace(get_repo=lambda repo: _FakeRepo())
    _patch_client(monkeypatch, fake)
    result = await gh.comment_on_pr("u", "o/r", 5, "looks good")
    assert result["id"] == 99 and result["url"] == "http://comment"


async def test_create_issue(monkeypatch):
    repo = _FakeRepo()
    fake = SimpleNamespace(get_repo=lambda r: repo)
    _patch_client(monkeypatch, fake)
    result = await gh.create_issue("u", "o/r", "Title", "Body")
    assert result["number"] == 7
    assert repo.created == [{"title": "Title", "body": "Body"}]


async def test_search_code(monkeypatch):
    hit = SimpleNamespace(repository=SimpleNamespace(full_name="o/r"),
                          path="a/b.py", html_url="http://code")
    fake = SimpleNamespace(search_code=lambda q: [hit])
    _patch_client(monkeypatch, fake)
    result = await gh.search_code("u", "def foo")
    assert result["results"][0]["path"] == "a/b.py"


async def test_list_recent_commits(monkeypatch):
    fake = SimpleNamespace(get_repo=lambda repo: _FakeRepo())
    _patch_client(monkeypatch, fake)
    result = await gh.list_recent_commits("u", "o/r", limit=5)
    assert result["commits"][0]["sha"] == "abc"
    assert result["commits"][0]["author"] == "me"
