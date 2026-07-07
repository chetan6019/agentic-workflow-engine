"""MCP GitHub server — read-heavy adapter on port 7007.

Per-user OAuth token (resolved from the integrations store) drives a PyGithub client. PyGithub
is synchronous, so each call runs in a worker thread via ``asyncio.to_thread`` to avoid blocking
the event loop. No delete_* tools; closing/renaming is HITL-gated through policy.yaml. The read
tools (list_my_prs, get_pr, search_code, list_recent_commits) are in policy.yaml's
trusted_read_actions, so the engine skips the response content-scan for them.
"""

from __future__ import annotations

import asyncio

import structlog
from fastmcp import FastMCP
from github import Auth, Github

from app.mcp.servers._shared import resolve_user_token

log = structlog.get_logger(__name__)
mcp = FastMCP("github")


def _client(token: str) -> Github:
    """Build a PyGithub client for one user's OAuth token."""
    return Github(auth=Auth.Token(token))


@mcp.tool()
async def list_my_prs(user_id: str, state: str = "open") -> dict:
    """List pull requests the authenticated user opened. Args: user_id (auto-injected), state."""
    token = await resolve_user_token(user_id, "github")

    def _fetch() -> list[dict]:
        client = _client(token)
        login = client.get_user().login
        found = client.search_issues(f"is:pr author:{login} state:{state}")
        rows: list[dict] = []
        for pr in found:
            rows.append({"repo": pr.repository.full_name, "number": pr.number,
                         "title": pr.title, "url": pr.html_url})
            if len(rows) >= 30:  # cap the page so a prolific author doesn't flood the result
                break
        return rows

    prs = await asyncio.to_thread(_fetch)
    log.info("github_list_my_prs", user_id=user_id, count=len(prs))
    return {"state": state, "pull_requests": prs}


@mcp.tool()
async def get_pr(user_id: str, repo: str, number: int) -> dict:
    """Get one pull request. Args: user_id (auto-injected), repo ("owner/name"), number."""
    token = await resolve_user_token(user_id, "github")

    def _fetch() -> dict:
        pr = _client(token).get_repo(repo).get_pull(number)
        return {"number": pr.number, "title": pr.title, "state": pr.state,
                "body": pr.body or "", "url": pr.html_url, "author": pr.user.login}

    data = await asyncio.to_thread(_fetch)
    log.info("github_get_pr", user_id=user_id, repo=repo, number=number)
    return data


@mcp.tool()
async def comment_on_pr(user_id: str, repo: str, number: int, body: str) -> dict:
    """Comment on a pull request. Args: user_id (auto-injected), repo, number, body.

    A write path: the body is body-size-capped by check_tool_args and secret-scanned by the
    output guardrails before this runs.
    """
    token = await resolve_user_token(user_id, "github")

    def _post() -> dict:
        comment = _client(token).get_repo(repo).get_issue(number).create_comment(body)
        return {"id": comment.id, "url": comment.html_url}

    data = await asyncio.to_thread(_post)
    log.info("github_comment_on_pr", user_id=user_id, repo=repo, number=number)
    return data


@mcp.tool()
async def create_issue(user_id: str, repo: str, title: str, body: str = "") -> dict:
    """Open an issue. Args: user_id (auto-injected), repo ("owner/name"), title, body."""
    token = await resolve_user_token(user_id, "github")

    def _create() -> dict:
        issue = _client(token).get_repo(repo).create_issue(title=title, body=body)
        return {"number": issue.number, "url": issue.html_url}

    data = await asyncio.to_thread(_create)
    log.info("github_create_issue", user_id=user_id, repo=repo)
    return data


@mcp.tool()
async def search_code(user_id: str, query: str) -> dict:
    """Search code across the user's accessible repos. Args: user_id (auto-injected), query."""
    token = await resolve_user_token(user_id, "github")

    def _search() -> list[dict]:
        found = _client(token).search_code(query)
        rows: list[dict] = []
        for hit in found:
            rows.append({"repo": hit.repository.full_name, "path": hit.path, "url": hit.html_url})
            if len(rows) >= 30:
                break
        return rows

    hits = await asyncio.to_thread(_search)
    log.info("github_search_code", user_id=user_id, count=len(hits))
    return {"query": query, "results": hits}


@mcp.tool()
async def list_recent_commits(user_id: str, repo: str, limit: int = 10) -> dict:
    """List recent commits on a repo's default branch. Args: user_id, repo, limit."""
    token = await resolve_user_token(user_id, "github")

    def _fetch() -> list[dict]:
        commits = _client(token).get_repo(repo).get_commits()
        out: list[dict] = []
        for commit in commits:
            out.append({"sha": commit.sha, "message": commit.commit.message,
                        "author": commit.commit.author.name, "url": commit.html_url})
            if len(out) >= limit:
                break
        return out

    rows = await asyncio.to_thread(_fetch)
    log.info("github_list_recent_commits", user_id=user_id, repo=repo, count=len(rows))
    return {"repo": repo, "commits": rows}


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=7007, path="/mcp")
