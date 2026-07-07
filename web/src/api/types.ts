// TypeScript mirrors of the backend Pydantic models (app/core/state.py + route models).

export interface PlanStep {
  id: string;
  tool: string;
  action: string;
  arguments: Record<string, unknown>;
  depends_on: string[];
}

export interface ExecutionPlan {
  reasoning: string;
  steps: PlanStep[];
  strategy: string;
  complexity_score: number;
  estimated_cost_usd: number;
  requires_approval: boolean;
}

export interface DraftResponse {
  summary: string;
  detail_markdown: string;
  actions_taken: string[];
  actions_pending: string[];
  citations: string[];
}

export interface ToolResult {
  step_id: string;
  ok: boolean;
  output: unknown;
  error: string | null;
  latency_ms: number;
}

// Subset of AgentState the UI actually reads (the API returns the full dict).
export interface AgentState {
  trace_id: string;
  session_id: string;
  user_request: string;
  plan: ExecutionPlan | null;
  tool_results: ToolResult[];
  draft: DraftResponse | null;
  confidence: number;
  requires_approval: boolean;
  approval_token: string | null;
  error: string | null;
  verdict: string | null;
  degraded: string[];
}

export interface RunResult {
  status: "pending" | "done" | "failed";
  state?: AgentState;
}

export interface PhaseEvent {
  phase: string;
  done: boolean;
}

export interface Session {
  id: string;
  title: string;
  status: string;
  last_activity: string;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface Approval {
  token: string;
  trace_id: string;
  expires_at: string;
}

export interface Integration {
  provider: string;
  status: "valid" | "expiring" | "revoked";
}
