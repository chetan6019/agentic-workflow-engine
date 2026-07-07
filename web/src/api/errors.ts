// Workflow error code → friendly markdown message.
// Port of streamlit_app/app.py::_error_message — keep the two in sync until
// Streamlit is retired.

export function errorMessage(err: string): string {
  if (err.startsWith("missing_integration:") || err.startsWith("no_token:")) {
    const tool = err.split(":", 2)[1];
    const cap = tool.charAt(0).toUpperCase() + tool.slice(1);
    return (
      `⚠️ **${cap} isn't connected.** Open the **Integrations** page, ` +
      `choose \`${tool}\`, add an access token, then send your request again.`
    );
  }
  if (err.startsWith("input_guardrail:")) {
    const rules = err.slice("input_guardrail:".length);
    return (
      `🛡️ **Request blocked by an input guardrail** (\`${rules}\`). ` +
      "Rephrase without instruction-override/jailbreak phrasing or " +
      "secret-shaped tokens, then try again."
    );
  }
  if (err.startsWith("guardrail_output:")) {
    const rules = err.slice("guardrail_output:".length);
    return (
      `🛡️ **Response blocked by an output guardrail** (\`${rules}\`) to ` +
      "avoid leaking sensitive data. Nothing was sent."
    );
  }
  if (err === "rate_limited") {
    return "⏳ **You're sending requests too quickly.** Wait a moment and retry.";
  }
  if (err === "pii_detected") {
    return "🛑 The draft contained sensitive data (PII) and was blocked.";
  }
  if (err === "low_confidence_blocked") {
    return (
      "🛑 I couldn't build a confident plan after retrying. " +
      "Try rephrasing or adding detail."
    );
  }
  if (err === "rejected_by_user") {
    return "❌ This workflow was rejected.";
  }
  if (err === "instance_restarted") {
    return "⚠️ The server restarted mid-run. Please send your request again.";
  }
  return `⚠️ Workflow error: \`${err}\``;
}

// Deterministic assistant text from the final workflow state.
// Port of streamlit_app/app.py::_final_text.
import type { AgentState } from "./types";

export function finalText(state: AgentState | null | undefined): string {
  if (!state) {
    return "⚠️ The workflow ended without a result — check the API logs.";
  }
  if (state.error) return errorMessage(state.error);
  const draft = state.draft;
  const md = (draft?.detail_markdown ?? "").trim();
  const summary = (draft?.summary ?? "").trim();
  const taken = (draft?.actions_taken ?? []).filter(Boolean).map((a) => `- ${a}`);
  const pending = (draft?.actions_pending ?? []).filter(Boolean).map((a) => `- ${a}`);
  // Primary body: composed reply, else summary, else the concrete actions taken.
  const body = md || summary || (taken.length ? "**Done:**\n" + taken.join("\n") : "");
  if (!body && !pending.length) {
    return "⚠️ The workflow finished but produced no draft content.";
  }
  const blocks = body ? [body] : [];
  if (pending.length) {
    blocks.push("**⏳ Needs follow-up:**\n" + pending.join("\n"));
  }
  return blocks.join("\n\n");
}
