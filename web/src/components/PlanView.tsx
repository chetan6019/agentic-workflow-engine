// Read-only table of the executed plan's steps (shown under a finished answer).

import type { ExecutionPlan } from "../api/types";

export function PlanView({ plan }: { plan: ExecutionPlan | null }) {
  if (!plan || plan.steps.length === 0) return null;
  return (
    <details className="plan-view">
      <summary>Plan — {plan.reasoning}</summary>
      <table>
        <thead>
          <tr><th>Step</th><th>Tool</th><th>Action</th><th>Depends on</th></tr>
        </thead>
        <tbody>
          {plan.steps.map((s) => (
            <tr key={s.id}>
              <td>{s.id}</td>
              <td>{s.tool}</td>
              <td>{s.action}</td>
              <td>{s.depends_on.join(", ") || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}
