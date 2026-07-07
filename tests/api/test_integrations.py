"""Async route tests for the integrations endpoints (list/connect/disconnect)."""

from __future__ import annotations

from app.security.crypto import pack_token


async def test_list_reports_provider_status(client, backend, auth_headers):
    backend.add_token("user-1", "gmail", pack_token("a"))                 # no expiry → valid
    backend.add_token("user-1", "calendar", pack_token("b", expires_in=60))  # ≤5min → expiring
    resp = await client.get("/v1/integrations", headers=auth_headers("user-1"))
    assert resp.status_code == 200
    status = {i["provider"]: i["status"] for i in resp.json()["integrations"]}
    assert status == {"gmail": "valid", "calendar": "expiring"}


async def test_list_is_user_scoped(client, backend, auth_headers):
    backend.add_token("owner-2", "gmail", pack_token("a"))
    resp = await client.get("/v1/integrations", headers=auth_headers("user-1"))
    assert resp.json()["integrations"] == []


async def test_connect_stores_encrypted_token(client, backend, auth_headers):
    resp = await client.post("/v1/integrations/slack/token", json={"token": "xoxb-secret"},
                             headers=auth_headers("user-1"))
    assert resp.status_code == 200 and resp.json()["provider"] == "slack"
    assert ("user-1", "slack") in backend.tokens
    assert "xoxb-secret" not in backend.tokens[("user-1", "slack")]  # stored encrypted


async def test_connect_unknown_provider_is_400(client, backend, auth_headers):
    resp = await client.post("/v1/integrations/dropbox/token", json={"token": "x"},
                             headers=auth_headers("user-1"))
    assert resp.status_code == 400


async def test_disconnect_deletes_token(client, backend, auth_headers):
    backend.add_token("user-1", "notion", pack_token("a"))
    resp = await client.request("DELETE", "/v1/integrations/notion", headers=auth_headers("user-1"))
    assert resp.status_code == 200 and resp.json()["status"] == "disconnected"
    assert ("user-1", "notion") not in backend.tokens


async def test_disconnect_missing_is_404(client, backend, auth_headers):
    resp = await client.request("DELETE", "/v1/integrations/gmail", headers=auth_headers("user-1"))
    assert resp.status_code == 404


async def test_disconnect_other_users_token_is_404(client, backend, auth_headers):
    backend.add_token("owner-2", "gmail", pack_token("a"))
    resp = await client.request("DELETE", "/v1/integrations/gmail", headers=auth_headers("user-1"))
    assert resp.status_code == 404
    assert ("owner-2", "gmail") in backend.tokens  # untouched


async def test_integrations_require_auth(client, backend):
    assert (await client.get("/v1/integrations")).status_code == 401
