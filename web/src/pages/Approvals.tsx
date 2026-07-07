// Approval inbox: pending approvals polled every 5s (matches the Streamlit
// fragment cadence); each row loads its run state to render the full card.

import { useEffect, useState } from "react";

import * as api from "../api/endpoints";
import type { AgentState, Approval } from "../api/types";
import { ApprovalCard } from "../components/ApprovalCard";

const POLL_MS = 5000;

export function Approvals() {
  const [pending, setPending] = useState<Approval[]>([]);
  const [openState, setOpenState] = useState<AgentState | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const r = await api.pendingApprovals();
        if (!cancelled) setPending(r.approvals);
      } catch {
        /* transient — next tick retries */
      }
    }
    tick();
    const id = window.setInterval(tick, POLL_MS);
    return () => { cancelled = true; window.clearInterval(id); };
  }, []);

  async function open(approval: Approval) {
    const r = await api.result(approval.trace_id);
    const state = (r.state ?? null) as AgentState | null;
    if (state) setOpenState({ ...state, approval_token: approval.token });
  }

  return (
    <div className="approvals-page">
      <h2>Approval inbox</h2>
      {pending.length === 0 && <p className="muted">Nothing waiting for approval.</p>}
      {pending.map((a) => (
        <div key={a.token} className="card session-row">
          <span>run <code>{a.trace_id.slice(0, 8)}…</code></span>
          <span className="muted">expires {new Date(a.expires_at).toLocaleString()}</span>
          <button onClick={() => open(a)}>Review</button>
        </div>
      ))}
      {openState?.approval_token && (
        <ApprovalCard token={openState.approval_token}
                      plan={openState.plan}
                      draft={openState.draft}
                      onDecided={() => setOpenState(null)} />
      )}
    </div>
  );
}
