// Approval card for a paused HITL run: plan-stage (review proposed steps before
// tools run) or draft-stage (review/edit the composed reply). Field labels and
// date formatting ported from streamlit_app/app.py::_render_pending_step.
// Steps outside the labeled fields (reddit.submit, github.close_pr, ...) render
// generically via the "Other" bucket, so every gated write is fully visible.

import { useState } from "react";

import * as api from "../api/endpoints";
import type { DraftResponse, ExecutionPlan, PlanStep } from "../api/types";
import { Markdown } from "./Markdown";

const STEP_FIELDS: Array<[string, string]> = [
  ["to", "📧 To"], ["cc", "Cc"], ["subject", "Subject"], ["body", "Body"],
  ["summary", "📅 Title"], ["start", "Start"], ["end", "End"],
  ["match_summary", "Event"], ["event_id", "Event id"],
  ["thread_id", "Thread id"], ["message_id", "Message id"],
  ["repo", "Repo"], ["number", "PR / Issue #"], ["title", "Title"],
  ["subreddit", "Subreddit"], ["parent_id", "Reply to"], ["query", "Query"],
];
const DT_FIELDS = new Set(["start", "end", "time_min", "time_max", "post_at"]);

function fmtDt(value: unknown): string {
  const d = new Date(String(value));
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString(undefined, {
    weekday: "short", day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

function isEmpty(v: unknown): boolean {
  return v === null || v === undefined || v === "" ||
         (Array.isArray(v) && v.length === 0);
}

function StepCard({ step }: { step: PlanStep }) {
  const args = step.arguments ?? {};
  const shown = new Set<string>();
  const rows = STEP_FIELDS.flatMap(([key, label]) => {
    const val = args[key];
    if (isEmpty(val)) return [];
    shown.add(key);
    return [[label, DT_FIELDS.has(key) ? fmtDt(val) : String(val)] as [string, string]];
  });
  const extra = Object.fromEntries(
    Object.entries(args).filter(([k, v]) => !shown.has(k) && !isEmpty(v)));

  return (
    <div className="card step-card">
      <div className="step-title">
        <code>{step.tool}.{step.action}</code> · step <code>{step.id}</code>
      </div>
      {rows.map(([label, text]) => (
        <div key={label}><strong>{label}:</strong> {text}</div>
      ))}
      {Object.keys(extra).length > 0 && (
        <div className="muted">Other: {JSON.stringify(extra)}</div>
      )}
      {rows.length === 0 && Object.keys(extra).length === 0 && (
        <div className="muted">No arguments.</div>
      )}
    </div>
  );
}

interface Props {
  token: string;
  plan: ExecutionPlan | null;
  draft: DraftResponse | null;
  onDecided: (decision: "approve" | "edit" | "reject") => void;
}

export function ApprovalCard({ token, plan, draft, onDecided }: Props) {
  const [edited, setEdited] = useState(draft?.detail_markdown ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const draftStage = !!draft?.detail_markdown || !!draft?.summary;

  async function decide(decision: "approve" | "edit" | "reject") {
    setBusy(true);
    setError(null);
    try {
      const editedDraft = decision === "edit" && draft
        ? { ...draft, detail_markdown: edited }
        : undefined;
      await api.decideApproval(token, decision, editedDraft);
      onDecided(decision);
    } catch (e) {
      // A timeout doesn't mean the resume failed — the caller re-fetches /result.
      setError(e instanceof Error ? e.message : String(e));
      onDecided(decision);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card approval-card">
      <h3>🛑 Approval required</h3>
      {draftStage && draft ? (
        <>
          <strong>{draft.summary}</strong>
          <Markdown text={draft.detail_markdown} />
          <textarea value={edited} onChange={(e) => setEdited(e.target.value)}
                    rows={8} disabled={busy} />
        </>
      ) : (
        <>
          <p><strong>Proposed plan</strong> — {plan?.reasoning}</p>
          <p className="muted">
            {plan?.steps.length ?? 0} action(s) will run after you approve:
          </p>
          {plan?.steps.map((s) => <StepCard key={s.id} step={s} />)}
        </>
      )}
      {error && <p className="error-text">{error}</p>}
      <div className="button-row">
        <button disabled={busy} onClick={() => decide("approve")}>
          {draftStage ? "✅ Approve" : "✅ Approve & run"}
        </button>
        <button disabled={busy || !draftStage} onClick={() => decide("edit")}>
          ✏️ Save edit
        </button>
        <button disabled={busy} onClick={() => decide("reject")}>❌ Reject</button>
      </div>
      {busy && <p className="muted">Resuming workflow… this can take a few minutes.</p>}
    </div>
  );
}
