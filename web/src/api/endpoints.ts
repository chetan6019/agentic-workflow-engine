// Typed wrappers for every backend endpoint the UI uses.

import { request } from "./client";
import type {
  Approval, DraftResponse, Integration, Message, PhaseEvent, RunResult, Session,
} from "./types";

// ── auth ─────────────────────────────────────────────────────────────────────

export function register(username: string, password: string) {
  return request<{ user_id: string; access_token: string }>(
    "POST", "/v1/auth/register", { body: { username, password } });
}

export function login(username: string, password: string) {
  return request<{ access_token: string }>(
    "POST", "/v1/auth/login", { body: { username, password } });
}

export function me() {
  return request<{ user_id: string; username: string;
                   integrations: Record<string, boolean> }>("GET", "/v1/auth/me");
}

// ── runs ─────────────────────────────────────────────────────────────────────

export function invoke(userRequest: string, sessionId: string | null,
                       requireApproval: boolean) {
  return request<{ trace_id: string }>("POST", "/v1/invoke", {
    body: { session_id: sessionId, user_request: userRequest,
            require_approval: requireApproval },
    // One key per send: a network retry replays the same run instead of duplicating it.
    headers: { "Idempotency-Key": crypto.randomUUID() },
    timeoutMs: 15_000,
  });
}

export function phase(traceId: string) {
  return request<PhaseEvent>("GET", `/v1/invoke/phase/${traceId}`,
                             { timeoutMs: 10_000 });
}

export function result(traceId: string) {
  return request<RunResult>("GET", `/v1/invoke/result/${traceId}`);
}

// ── approvals ────────────────────────────────────────────────────────────────

export function pendingApprovals() {
  return request<{ approvals: Approval[] }>("GET", "/v1/approvals?status=pending");
}

export function decideApproval(token: string, decision: "approve" | "edit" | "reject",
                               editedDraft?: DraftResponse) {
  return request<{ status: string }>("POST", `/v1/approvals/${token}`, {
    body: { decision, edited_draft: editedDraft ?? null },
    // The resume runs the graph synchronously — mirror Streamlit's 180s budget.
    timeoutMs: 180_000,
  });
}

// ── sessions ─────────────────────────────────────────────────────────────────

export function listSessions() {
  return request<{ sessions: Session[] }>("GET", "/v1/sessions");
}

export function getSession(id: string) {
  return request<{ id: string; title: string; messages: Message[] }>(
    "GET", `/v1/sessions/${id}`);
}

export function renameSession(id: string, title: string) {
  return request<{ status: string; title: string }>(
    "PATCH", `/v1/sessions/${id}`, { body: { title } });
}

// ── integrations ─────────────────────────────────────────────────────────────

export function listIntegrations() {
  return request<{ integrations: Integration[] }>("GET", "/v1/integrations");
}

export function saveIntegrationToken(provider: string, token: string,
                                     refreshToken?: string, expiresIn?: number) {
  return request<{ status: string; provider: string }>(
    "POST", `/v1/integrations/${provider}/token`,
    { body: { token, refresh_token: refreshToken || null,
              expires_in: expiresIn ?? null } });
}

export function deleteIntegration(provider: string) {
  return request<{ status: string; provider: string }>(
    "DELETE", `/v1/integrations/${provider}`);
}

// ── feedback ─────────────────────────────────────────────────────────────────

export function sendFeedback(traceId: string, score: number, comment?: string) {
  return request<{ status: string }>("POST", `/v1/feedback/${traceId}`,
                                     { body: { score, comment: comment || null } });
}
