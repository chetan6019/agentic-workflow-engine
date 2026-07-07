// Structured, row-per-detail rendering of a finished run: the answer body plus
// a labeled table (summary, actions, follow-ups, citations, confidence) instead
// of one undifferentiated markdown blob.

import { errorMessage } from "../api/errors";
import type { AgentState } from "../api/types";
import { Markdown } from "./Markdown";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="detail-row">
      <div className="detail-label">{label}</div>
      <div className="detail-value">{children}</div>
    </div>
  );
}

export function ResultDetails({ state }: { state: AgentState }) {
  if (state.error) {
    return <Markdown text={errorMessage(state.error)} />;
  }
  const draft = state.draft;
  const md = (draft?.detail_markdown ?? "").trim();
  const summary = (draft?.summary ?? "").trim();
  const taken = (draft?.actions_taken ?? []).filter(Boolean);
  const pending = (draft?.actions_pending ?? []).filter(Boolean);
  // Only URLs are meaningful to users; older runs sometimes have internal step
  // ids ("s1") in citations — hide those (the composer prompt now forbids them).
  const citations = (draft?.citations ?? []).filter((c) => /^https?:\/\//.test(c));

  if (!md && !summary && !taken.length && !pending.length) {
    return <Markdown text="⚠️ The workflow finished but produced no draft content." />;
  }

  return (
    <div className="result-details">
      {md && <Markdown text={md} />}
      <div className="detail-table">
        {summary && <Row label="Summary">{summary}</Row>}
        {taken.length > 0 && (
          <Row label="✅ Actions taken">
            <ul>{taken.map((a, i) => <li key={i}>{a}</li>)}</ul>
          </Row>
        )}
        {pending.length > 0 && (
          <Row label="⏳ Follow-ups">
            <ul>{pending.map((a, i) => <li key={i}>{a}</li>)}</ul>
          </Row>
        )}
        {citations.length > 0 && (
          <Row label="📎 Citations">
            <ul>{citations.map((c, i) => <li key={i}><code>{c}</code></li>)}</ul>
          </Row>
        )}
        {state.confidence > 0 && (
          <Row label="Confidence">
            <span className={`pill pill-${state.confidence >= 0.85 ? "green" : "amber"}`}>
              {(state.confidence * 100).toFixed(0)}%
            </span>
          </Row>
        )}
        {state.degraded?.length > 0 && (
          <Row label="⚠️ Degraded">{state.degraded.join(", ")}</Row>
        )}
      </div>
    </div>
  );
}
