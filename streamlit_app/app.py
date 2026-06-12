"""Streamlit UI — login, chat, plan inspector, history, HITL approval."""
from __future__ import annotations

import json
import os
import time
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
<div class="cap-card"><div class="icon">📅</div>
<div class="title">Calendar</div><div class="desc">Schedule, move, or cancel events</div></div>
<div class="cap-card"><div class="icon">📧</div>
<div class="title">Gmail</div><div class="desc">Draft, search, and send emails</div></div>
<div class="cap-card"><div class="icon">📝</div>
<div class="title">Notion</div><div class="desc">Create pages, update databases</div></div>
<div class="cap-card"><div class="icon">💬</div>
<div class="title">Slack</div><div class="desc">Post messages, search channels</div></div>
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


def _integration_form() -> None:
    """Expander to submit an access/refresh token for a provider."""
    flash = st.session_state.pop("int_flash", None)
    if flash:
        st.success(flash)
    with st.expander("🔌 Connect an integration"):
        provider = st.selectbox(
            "Provider", ["gmail", "calendar", "notion", "slack"], key="int_provider")
        is_google = provider in ("gmail", "calendar")
        with st.form("f_integration"):
            token = st.text_input("Access token", type="password")
            refresh = (st.text_input("Refresh token (optional)", type="password")
                       if is_google else "")
            expires_in = (st.number_input(
                "Expires in (seconds, optional)", min_value=0, value=0, step=60)
                if is_google else 0)
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
        st.markdown("**Integrations**")
        st.caption("Green = connected. Use 🔌 below to connect a provider.")
        r = _call("GET", "/v1/auth/me")
        me = r.json() if r and r.status_code == 200 else {}
        ints = me.get("integrations", {}) if isinstance(me, dict) else {}
        c1, c2 = st.columns(2)
        c1.markdown(status_pill("calendar", "green" if ints.get("calendar") else "slate")
                    + " " + status_pill("gmail", "green" if ints.get("gmail") else "slate"),
                    unsafe_allow_html=True)
        c2.markdown(status_pill("notion", "green" if ints.get("notion") else "slate")
                    + " " + status_pill("slack", "green" if ints.get("slack") else "slate"),
                    unsafe_allow_html=True)
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
    if md:
        return md
    # Composer occasionally leaves detail_markdown empty for single-action
    # confirmations; fall back to the summary, then the actions taken.
    summary = (draft.get("summary") or "").strip()
    if summary:
        return summary
    taken = [f"- {a}" for a in (draft.get("actions_taken") or [])]
    if taken:
        return "**Done:**\n" + "\n".join(taken)
    return "⚠️ The workflow finished but produced no draft content."


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
        r = _call("GET", f"/v1/invoke/phase/{trace_id}", timeout=5)
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
    else:
        st.session_state.messages.append(
            {"role": "assistant", "content": r.text if r else "⚠️ API unavailable."})
    st.rerun()


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
            if steps:
                st.dataframe([{
                    "id": s.get("id"), "tool": s.get("tool"), "action": s.get("action"),
                    "to": (s.get("arguments") or {}).get("to", ""),
                    "subject": (s.get("arguments") or {}).get("subject", ""),
                    "arguments": json.dumps(s.get("arguments", {})),
                } for s in steps], width="stretch")
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
               "(Gmail, Calendar, Notion, Slack), and draft a reply.", text_alignment="center")
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
