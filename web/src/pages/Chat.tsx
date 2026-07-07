// Main chat page: prompt box → POST /v1/invoke → live SSE progress → answer.
// Each assistant answer renders structured (ResultDetails rows); the executed
// plan is deliberately NOT shown in responses. Transcript state lives in
// ChatStore so navigating to other pages and back keeps the conversation.
// A paused run (requires_approval) renders the ApprovalCard inline.

import { useEffect, useRef, useState } from "react";

import * as api from "../api/endpoints";
import { errorMessage } from "../api/errors";
import type { AgentState } from "../api/types";
import { useChatStore } from "../chat/ChatStore";
import { ApprovalCard } from "../components/ApprovalCard";
import { FeedbackWidget } from "../components/FeedbackWidget";
import { Markdown } from "../components/Markdown";
import { PhasePill } from "../components/PhasePill";
import { ResultDetails } from "../components/ResultDetails";
import { useRunStream } from "../hooks/useRunStream";

export function Chat() {
  const { entries, setEntries, traceId, setTraceId,
          pendingApproval, setPendingApproval,
          requireApproval, setRequireApproval } = useChatStore();
  const [prompt, setPrompt] = useState("");
  const [sendError, setSendError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // Navigating away closes the EventSource (unmount); coming back re-connects
  // with the same trace id — the stream endpoint replays a finished run's
  // result immediately, so nothing is lost mid-run.
  const { phase, result } = useRunStream(traceId);

  // Terminal result for the ACTIVE run only (the hook tags results by trace id,
  // so a new query can never surface the previous query's answer).
  useEffect(() => {
    if (!result || !traceId) return;
    const state = (result.state ?? null) as AgentState | null;
    if (state?.requires_approval && state.approval_token && !state.error) {
      setPendingApproval(state);
      return;
    }
    setEntries((es) => [...es, { role: "assistant", state }]);
    setTraceId(null);
  }, [result, traceId, setEntries, setPendingApproval, setTraceId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries, traceId, pendingApproval]);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const text = prompt.trim();
    if (text.length < 4 || traceId) return;
    setSendError(null);
    setPendingApproval(null);
    setEntries((es) => [...es, { role: "user", text }]);
    setPrompt("");
    try {
      const resp = await api.invoke(text, null, requireApproval);
      setTraceId(resp.trace_id);
    } catch (err) {
      setSendError(err instanceof Error ? err.message : String(err));
    }
  }

  async function onApprovalDecided(decision: "approve" | "edit" | "reject") {
    const tid = pendingApproval?.trace_id;
    setPendingApproval(null);
    setTraceId(null);
    if (decision === "reject") {
      setEntries((es) => [...es, { role: "assistant",
                                   text: errorMessage("rejected_by_user") }]);
      return;
    }
    // The resume already ran (synchronously); the persisted result is authoritative.
    if (tid) {
      const r = await api.result(tid);
      setEntries((es) => [...es, { role: "assistant",
                                   state: (r.state ?? null) as AgentState | null }]);
    }
  }

  return (
    <div className="chat-page">
      <div className="chat-log">
        {entries.length === 0 && !traceId && (
          <div className="empty-hint">
            <div className="empty-icon">⚙️</div>
            <h3>What can I run for you?</h3>
            <p className="muted">
              Check unread mail, book meetings, review PRs, scan subreddits,
              look up stock quotes — all through one prompt.
            </p>
          </div>
        )}
        {entries.map((entry, i) => (
          <div key={i} className={`msg ${entry.role}`}>
            <div className="avatar">{entry.role === "user" ? "🧑" : "🤖"}</div>
            <div className="bubble">
              {entry.state !== undefined
                ? entry.state
                  ? (
                    <>
                      <ResultDetails state={entry.state} />
                      {!entry.state.error && (
                        <FeedbackWidget traceId={entry.state.trace_id} />
                      )}
                    </>
                  )
                  : <Markdown text="⚠️ The workflow ended without a result — check the API logs." />
                : <Markdown text={entry.text ?? ""} />}
            </div>
          </div>
        ))}
        {traceId && !pendingApproval && (
          <div className="msg assistant">
            <div className="avatar">🤖</div>
            <div className="bubble working"><PhasePill label={phase} pulse /></div>
          </div>
        )}
        {pendingApproval?.approval_token && (
          <ApprovalCard token={pendingApproval.approval_token}
                        plan={pendingApproval.plan}
                        draft={pendingApproval.draft}
                        onDecided={onApprovalDecided} />
        )}
        {sendError && <p className="error-text">{sendError}</p>}
        <div ref={bottomRef} />
      </div>
      <form className="chat-input" onSubmit={send}>
        <input value={prompt} placeholder="Ask me to check mail, book meetings, look at PRs…"
               onChange={(e) => setPrompt(e.target.value)} disabled={!!traceId} />
        <label className="muted">
          <input type="checkbox" checked={requireApproval}
                 onChange={(e) => setRequireApproval(e.target.checked)} />
          require approval
        </label>
        <button type="submit" disabled={!!traceId || prompt.trim().length < 4}>
          Send ➤
        </button>
      </form>
    </div>
  );
}
