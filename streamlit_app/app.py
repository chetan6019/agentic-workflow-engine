"""Streamlit UI — login, chat, plan inspector, history, HITL approval."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any

import httpx
import streamlit as st
from styles import inject_styles, status_pill

API = os.environ.get("API_BASE_URL", "http://localhost:8000")
_INIT: dict[str, Any] = {
    "access_token": None, "user_id": None, "session_id": None, "messages": [],
    "current_trace_id": None, "pending_approval": None, "last_state": None,
    "require_approval": False, "pending_run": None,
}
_WELCOME = """<div class="welcome-hero">
<h2>What can I help you with?</h2>
<p>Describe a task and I'll plan it, execute tool calls, and draft a response pausing for your approval when needed.</p>
<div class="cap-grid">
<div class="cap-card"><div class="icon">📬</div>
<div class="title">Google</div><div class="desc">Gmail + Calendar: search, draft, send, schedule</div></div>
<div class="cap-card"><div class="icon">🐙</div>
<div class="title">GitHub</div><div class="desc">PRs, issues, code search, recent commits</div></div>
<div class="cap-card"><div class="icon">👽</div>
<div class="title">Reddit</div><div class="desc">Search, read posts, comment (with approval)</div></div>
<div class="cap-card"><div class="icon">📈</div>
<div class="title">Finnhub</div><div class="desc">Quotes, profiles, news — market data (read-only)</div></div>
</div></div>"""


def _call(method: str, path: str, *, timeout: int = 30, **kw: Any) -> httpx.Response | None:
    """Sync API call with auth; returns None on connection failure."""
    tok = st.session_state.get("access_token")
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    try:
        return httpx.request(method, f"{API}{path}", headers=hdr, timeout=timeout, **kw)
    except httpx.ConnectError:
        st.error("Cannot reach the API server.")
        return None
    except httpx.TimeoutException:
        # A slow tick (e.g. first request while the embedder is still warming) is
        # transient: return None so the caller's poll loop just retries next tick
        # instead of crashing the page with an httpx.ReadTimeout traceback.
        return None


def _detail(r: httpx.Response) -> str:
    """Pull the API's `detail` field from a JSON error body, falling back to text."""
    try:
        return str(r.json().get("detail", r.text))
    except Exception:
        return r.text


# Canonical user-facing providers shown in the sidebar panel, in display order.
_PANEL_PROVIDERS = ("google", "github", "reddit")
# Legacy single-service token keys now subsumed by the combined "google" provider.
_LEGACY_GOOGLE = {"gmail", "calendar"}


def _canon_provider(provider: str) -> str:
    """Map a stored provider key to its canonical user-facing provider."""
    return "google" if provider in _LEGACY_GOOGLE else provider


def _integrations_panel() -> None:
    """Show canonical providers (google/github/reddit) with token health + disconnect.

    The API returns rows keyed by the *stored* provider, which may include legacy
    gmail/calendar tokens — those are folded into "google" here so the panel reflects
    the current provider model rather than raw storage keys.
    """
    st.markdown("**Integrations**")
    st.caption("Connected providers and token health. Use 🔌 below to add one.")
    r = _call("GET", "/v1/integrations")
    items = ((r.json() or {}).get("integrations", [])
             if r and r.status_code == 200 else [])
    status_by: dict[str, str] = {}
    raw_by: dict[str, list[str]] = {}
    for it in items:
        canon = _canon_provider(it.get("provider", ""))
        raw_by.setdefault(canon, []).append(it.get("provider", ""))
        status_by.setdefault(canon, it.get("status", "valid"))
    colors = {"valid": "green", "expiring": "amber", "revoked": "red"}
    for prov in _PANEL_PROVIDERS:
        status = status_by.get(prov)
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        c1.markdown(
            status_pill(f"{prov} · {status or 'not connected'}",
                        colors.get(status or "", "slate")),
            unsafe_allow_html=True)
        if status and c2.button("✕", key=f"disc_{prov}", help=f"Disconnect {prov}"):
            # "google" may be backed by legacy gmail/calendar rows — delete each.
            ok = False
            for raw in raw_by.get(prov, [prov]):
                dr = _call("DELETE", f"/v1/integrations/{raw}")
                ok = ok or (dr is not None and dr.status_code == 200)
            if ok:
                st.session_state["int_flash"] = f"🔌 {prov} disconnected."
                st.rerun()
            else:
                st.error(f"Could not disconnect {prov}.")


def _integration_form() -> None:
    """Expander to submit an access/refresh token for a provider."""
    flash = st.session_state.pop("int_flash", None)
    if flash:
        st.success(flash)
    with st.expander("🔌 Connect an integration"):
        # Per-user OAuth providers only. Finnhub is read-only via a shared API key, so
        # it needs no per-user connection and is intentionally absent from the sidebar.
        provider = st.selectbox(
            "Provider", ["google", "reddit", "github"], key="int_provider")
        with st.form("f_integration"):
            token = st.text_input("Access token", type="password")
            refresh = st.text_input("Refresh token (optional)", type="password")
            expires_in = st.number_input(
                "Expires in (seconds, optional)", min_value=0, value=0, step=60)
            if st.form_submit_button("Save token", width=True):
                if not token:
                    st.warning("Access token is required.")
                    return
                body: dict[str, Any] = {"token": token}
                if refresh:
                    body["refresh_token"] = refresh
                if expires_in:
                    body["expires_in"] = int(expires_in)
                r = _call("POST", f"/v1/integrations/{provider}/token", json=body)
                if r and r.status_code == 200:
                    st.session_state["int_flash"] = f"✅ {provider} connected."
                    st.rerun()
                elif r:
                    st.error(r.text)


def _auth_forms() -> None:
    """Login / register tabs shown in the sidebar when signed out."""
    st.markdown("### 🔐 Sign in")
    for tab, label, path in zip(
            st.tabs(["Login", "Register"]), ["Login", "Register"],
            ["/v1/auth/login", "/v1/auth/register"]):
        with tab, st.form(f"f_{path}"):
            u = st.text_input("Username", key=f"u_{path}")
            p = st.text_input("Password", type="password", key=f"p_{path}")
            if st.form_submit_button(label, width="stretch"):
                r = _call("POST", path, json={"username": u, "password": p})
                if r and r.status_code == 200:
                    st.session_state.update(**r.json())
                    st.rerun()
                elif r:
                    try:
                        msg = r.json().get("detail", r.text)
                    except Exception:
                        msg = r.text
                    st.error(msg)


def _session_picker() -> None:
    """Session dropdown (by title), rename control, and new-session button."""
    r = _call("GET", "/v1/sessions")
    sessions = ((r.json() or {}).get("sessions", [])
                if r and r.status_code == 200 else [])
    labels = {s["id"]: (s.get("title") or f"Session {s['id'][:8]}") for s in sessions}
    choice = st.selectbox(
        "Session", ["(new)"] + [s["id"] for s in sessions],
        format_func=lambda x: "➕ New session" if x == "(new)" else labels.get(x, x))
    st.session_state.session_id = None if choice == "(new)" else choice
    if st.session_state.session_id:
        with st.expander("✏️ Rename session"):
            name = st.text_input("Name", value=labels.get(st.session_state.session_id, ""),
                                 key="rename_input")
            if st.button("Save name", width="stretch") and name.strip():
                rr = _call("PATCH", f"/v1/sessions/{st.session_state.session_id}",
                           json={"title": name.strip()})
                if rr and rr.status_code == 200:
                    st.rerun()
                elif rr:
                    st.error(rr.text)
    if st.button("➕ New session", width="stretch"):
        st.session_state.update(session_id=None, messages=[])
        st.rerun()


def _sidebar() -> None:
    """Sidebar: auth, integration pills, session picker."""
    with st.sidebar:
        if not st.session_state.access_token:
            _auth_forms()
            return
        _integrations_panel()
        _integration_form()
        st.divider()
        st.markdown("**🛑 HITL approval**")
        st.caption("Pause after planning so you can review and approve before tools run.")
        st.toggle("Enable approval gate", key="require_approval",
                  help="When on, the agent waits for your **Approve & run** click "
                       "before executing — use it for sends, deletes, and edits.")
        st.divider()
        _session_picker()


def _error_message(err: str) -> str:
    """Map a workflow error code to a friendly, actionable chat message."""
    if err.startswith(("missing_integration:", "no_token:")):
        tool = err.split(":", 1)[1]
        return (f"⚠️ **{tool.capitalize()} isn't connected.** Open "
                f"**🔌 Connect an integration** in the sidebar, choose `{tool}`, "
                f"and add an access token then send your request again.")
    if err.startswith("input_guardrail:"):
        rules = err.split(":", 1)[1]
        return (f"🛡️ **Request blocked by an input guardrail** (`{rules}`). "
                "Rephrase without instruction-override/jailbreak phrasing or "
                "secret-shaped tokens, then try again.")
    if err.startswith("guardrail_output:"):
        rules = err.split(":", 1)[1]
        return (f"🛡️ **Response blocked by an output guardrail** (`{rules}`) to "
                "avoid leaking sensitive data. Nothing was sent.")
    if err == "rate_limited":
        return "⏳ **You're sending requests too quickly.** Wait a moment and retry."
    if err == "pii_detected":
        return "🛑 The draft contained sensitive data (PII) and was blocked."
    if err == "low_confidence_blocked":
        return ("🛑 I couldn't build a confident plan after retrying. "
                "Try rephrasing or adding detail.")
    if err == "rejected_by_user":
        return "❌ This workflow was rejected."
    return f"⚠️ Workflow error: `{err}`"


def _fetch_result(trace_id: str, tries: int = 30) -> dict[str, Any] | None:
    """Poll the persisted result endpoint authoritative source if SSE drops."""
    for _ in range(tries):
        r = _call("GET", f"/v1/invoke/result/{trace_id}")
        if r and r.status_code == 200 and r.json().get("status") == "done":
            return r.json().get("state")
        time.sleep(1)
    return None


def _final_text(final: dict[str, Any] | None) -> str:
    """Deterministic assistant text from the final workflow state."""
    if not final:
        return "⚠️ The workflow stream ended without a result check the API logs."
    if final.get("error"):
        return _error_message(final["error"])
    draft = final.get("draft") or {}
    md = (draft.get("detail_markdown") or "").strip()
    summary = (draft.get("summary") or "").strip()
    taken = [f"- {a}" for a in (draft.get("actions_taken") or []) if a]
    pending = [f"- {a}" for a in (draft.get("actions_pending") or []) if a]
    # Primary body: the composed reply, else the summary, else the concrete actions taken
    # (the composer sometimes leaves detail_markdown empty for single-action confirmations).
    body = md or summary or ("**Done:**\n" + "\n".join(taken) if taken else "")
    if not body and not pending:
        return "⚠️ The workflow finished but produced no draft content."
    blocks = [body] if body else []
    # Always surface follow-ups so something concrete appears after an approved action.
    if pending:
        blocks.append("**⏳ Needs follow-up:**\n" + "\n".join(pending))
    return "\n\n".join(blocks)


def _phase_color(label: str) -> str:
    """Pick a pill color for a phase label."""
    s = label.lower()
    if "fail" in s or "❌" in s:
        return "red"
    if "approval" in s or "🛑" in s:
        return "amber"
    if "done" in s or "finaliz" in s or "💾" in s:
        return "green"
    return "purple"


def _poll_run(trace_id: str) -> dict[str, Any] | None:
    """Poll /phase and re-render a single small pill in place as phases advance."""
    placeholder = st.empty()
    last = ""
    for _ in range(360):  # up to ~180s of progress
        r = _call("GET", f"/v1/invoke/phase/{trace_id}", timeout=10)
        if r and r.status_code == 200:
            data = r.json()
            label = data.get("phase") or "working"
            if label != last:
                placeholder.markdown(
                    status_pill(label, _phase_color(label)), unsafe_allow_html=True)
                last = label
            if data.get("done"):
                placeholder.markdown(
                    status_pill(f"done — {label}", "green"), unsafe_allow_html=True)
                break
        time.sleep(0.5)
    else:
        placeholder.markdown(
            status_pill("timed out", "red"), unsafe_allow_html=True)
        return None
    return _fetch_result(trace_id, tries=10)


def _render_chat() -> None:
    """Replay the conversation, drive any pending run's progress, show HITL."""
    if not st.session_state.messages:
        st.markdown(_WELCOME, unsafe_allow_html=True)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    pending = st.session_state.pop("pending_run", None)
    if pending:
        with st.chat_message("assistant"):
            final = _poll_run(pending)
        _store_result(final)
        st.rerun()
    _hitl_block()


def _store_result(final: dict[str, Any] | None) -> None:
    """Persist the assistant reply (and any approval) into session state."""
    text = _final_text(final)
    if final:
        st.session_state.last_state = final
    st.session_state.messages.append({"role": "assistant", "content": text})
    if final and final.get("requires_approval") and not final.get("error"):
        st.session_state.pending_approval = final


def _submit_prompt(prompt: str) -> None:
    """Kick the workflow off in the background, then rerun so _render_chat polls it."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    r = _call("POST", "/v1/invoke", timeout=15,
              json={"session_id": st.session_state.session_id,
                    "user_request": prompt,
                    "require_approval": bool(st.session_state.get("require_approval"))})
    if r and r.status_code == 200:
        trace_id = r.json().get("trace_id")
        st.session_state.current_trace_id = trace_id
        st.session_state.pending_run = trace_id
    elif r is not None and r.status_code in (400, 429):
        # Synchronous input-guardrail / rate-cap rejection from /v1/invoke.
        st.session_state.messages.append(
            {"role": "assistant", "content": _error_message(_detail(r))})
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": r.text if r else "⚠️ API unavailable."})
    st.rerun()


# Argument fields surfaced (in order) on an approval card, with friendly labels. Anything
# not listed is still shown under "Other", so the approver never acts on hidden data.
_STEP_FIELDS = (
    ("to", "📧 To"), ("cc", "Cc"), ("subject", "Subject"), ("body", "Body"),
    ("summary", "📅 Title"), ("start", "Start"), ("end", "End"),
    ("match_summary", "Event"), ("event_id", "Event id"),
    ("thread_id", "Thread id"), ("message_id", "Message id"),
    ("repo", "Repo"), ("number", "PR / Issue #"), ("title", "Title"),
    ("subreddit", "Subreddit"), ("parent_id", "Reply to"), ("query", "Query"),
)
_DT_FIELDS = {"start", "end", "time_min", "time_max", "post_at"}


def _fmt_dt(value: Any) -> str:
    """Render an ISO-8601 timestamp as 'Mon 23 Jun 2026, 14:30'; pass through on failure."""
    try:
        return datetime.fromisoformat(str(value)).strftime("%a %d %b %Y, %H:%M")
    except (ValueError, TypeError):
        return str(value)


def _render_pending_step(step: dict) -> None:
    """Render one proposed plan step as a readable approval card (To/Subject/Body, dates…)."""
    tool, action = step.get("tool", ""), step.get("action", "")
    args = step.get("arguments") or {}
    st.markdown(f"**`{tool}.{action}`**  ·  step `{step.get('id', '')}`")
    shown_keys: set[str] = set()
    for key, label in _STEP_FIELDS:
        val = args.get(key)
        if val in (None, "", []):
            continue
        shown_keys.add(key)
        text = _fmt_dt(val) if key in _DT_FIELDS else val
        st.markdown(f"- **{label}:** {text}")
    extra = {k: v for k, v in args.items()
             if k not in shown_keys and v not in (None, "", [])}
    if extra:
        st.caption("Other: " + json.dumps(extra, default=str))
    if not shown_keys and not extra:
        st.caption("No arguments.")


def _hitl_block() -> None:
    """Approve / edit / reject for a paused HITL workflow (plan-stage or draft-stage)."""
    p = st.session_state.pending_approval
    if not p:
        return
    token = p.get("approval_token")
    draft = p.get("draft") or {}
    plan = p.get("plan") or {}
    with st.container(border=True):
        st.subheader("🛑 Approval required")
        if draft:  # draft-stage: review the composed reply
            st.markdown(f"**{draft.get('summary', '')}**\n\n{draft.get('detail_markdown', '')}")
            edited = st.text_area("Edit draft",
                                  value=draft.get("detail_markdown", ""), height=200)
        else:  # plan-stage: review the proposed plan before tools run
            st.markdown(f"**Proposed plan** — {plan.get('reasoning', '')}")
            steps = plan.get("steps") or []
            st.caption(f"{len(steps)} action(s) will run after you approve:")
            for s in steps:
                with st.container(border=True):
                    _render_pending_step(s)
            edited = ""
        c1, c2, c3 = st.columns(3)
        label_ok = "✅ Approve" if draft else "✅ Approve & run"
        b1 = c1.button(label_ok, width="stretch")
        b2 = c2.button("✏️ Edit", width="stretch", disabled=not draft)
        b3 = c3.button("❌ Reject", width="stretch")
        dec = "approve" if b1 else "edit" if b2 else "reject" if b3 else None
        if not (dec and token):
            return
        ed = {**draft, "detail_markdown": edited} if dec == "edit" and draft else None
        with st.spinner("Resuming workflow…"):
            _call("POST", f"/v1/approvals/{token}", timeout=180,
                  json={"decision": dec, "edited_draft": ed})
        st.session_state.pending_approval = None
        if dec == "reject":
            _store_result({"error": "rejected_by_user"})
        else:
            _store_result(_fetch_result(st.session_state.current_trace_id))
        st.rerun()


def _plan_tab() -> None:
    """Plan JSON, confidence pill, and retrieved-plans table."""
    s = st.session_state.last_state or {}
    if not s.get("plan") and not s.get("retrieved_plans"):
        st.info("Run a request in **Chat** first. This tab then shows *how* the agent "
                "decided to fulfil it the step-by-step plan, its confidence, and the "
                "past workflows it learned from.")
        return
    conf = float(s.get("confidence", 0.0))
    color = "green" if conf >= 0.85 else ("amber" if conf >= 0.55 else "red")
    st.markdown(f"**Confidence:** {status_pill(f'{conf:.0%}', color)}", unsafe_allow_html=True)
    st.caption("How sure the agent is in the result blends tool success, similarity to "
               "past plans, and an LLM quality check. ≥85% auto-completes; lower can re-plan.")
    if s.get("plan"):
        st.markdown("**Execution plan** the ordered steps generated for your request:")
        st.json(s["plan"])
    if s.get("retrieved_plans"):
        st.markdown("**Similar past plans** earlier successful runs the retriever used "
                    "as guidance:")
        st.dataframe(
            [{"summary": rp.get("summary", ""),
              "similarity": f"{rp.get('similarity', 0.0):.2f}",
              "tools": ", ".join(
                  step.get("tool", "") for step in
                  (rp.get("plan_json") or {}).get("steps", []))}
             for rp in s["retrieved_plans"]], width="stretch")


def _history_tab() -> None:
    """Past sessions; select one to load its messages."""
    r = _call("GET", "/v1/sessions")
    data = ((r.json() or {}).get("sessions", [])
            if r and r.status_code == 200 else [])
    if not data:
        st.info("No prior sessions yet. Start a conversation to see history.")
        return
    for s in data:
        pill = status_pill(
            s.get("status", "?"),
            "green" if s.get("status") == "active" else "slate")
        c1, c2 = st.columns([4, 1])
        c1.markdown(
            f"{pill} **{s['id'][:8]}…** — {s.get('last_activity', '')}",
            unsafe_allow_html=True)
        if c2.button("Load", key=f"ld_{s['id']}"):
            r2 = _call("GET", f"/v1/sessions/{s['id']}")
            msgs = (r2.json().get("messages", [])
                    if r2 and r2.status_code == 200 else [])
            st.session_state.update(session_id=s["id"],
                messages=[{"role": m["role"], "content": m["content"]}
                          for m in msgs])
            st.rerun()


def main() -> None:
    """Entrypoint: page config, styles, sidebar, three tabs."""
    st.set_page_config(
        layout="wide", page_icon="🤖", page_title="Workflow Agent")
    inject_styles()
    for k, v in _INIT.items():
        st.session_state.setdefault(k, [] if isinstance(v, list) else v)
    _sidebar()
    if not st.session_state.access_token:
        st.markdown(
            '<div class="welcome-hero"><h2>Welcome to Workflow Agent</h2>'
            "<p>Log in from the sidebar to get started.</p></div>",
            unsafe_allow_html=True)
        return
    #st.markdown("## 🤖 Workflow Agent")
    st.title("🤖 Workflow Agent", text_alignment="center")
    st.caption("Describe a task in plain English.I plan it, run the tools "
               "(Google, GitHub, Reddit, Finnhub), and draft a reply.", text_alignment="center")
    chat, inspector, history = st.tabs(
        ["💬 Chat", "🔍 Plan Inspector", "📋 History"])
    with chat:
        st.caption("Ask the agent to do something; replies appear here and the input "
                   "box stays docked at the bottom of the screen.")
        _render_chat()
    with inspector:
        st.caption("See *how* the agent reasoned about your last request its plan, "
                   "confidence, and the past workflows it drew on.")
        _plan_tab()
    with history:
        st.caption("Your past sessions. Load one to revisit its conversation.")
        _history_tab()
    # Page-level chat input → Streamlit pins it to the bottom of the screen.
    if prompt := st.chat_input("Describe a task e.g. 'Show my unread emails'"):
        _submit_prompt(prompt)


main()
