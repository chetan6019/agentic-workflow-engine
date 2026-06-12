"""Streamlit CSS injection and UI primitive helpers."""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* ---- Base ---- */
[data-testid="stAppViewContainer"] { background: #0e1117; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161922 0%, #11131a 100%);
    border-right: 1px solid #23263a;
}
.block-container { padding: 2.5rem 2rem 1rem; max-width: 1100px; }

/* ---- Fix tab clipping at top ---- */
[data-testid="stTabs"] { margin-top: 0.25rem; }
[data-testid="stTabs"] button {
    font-size: 0.95rem;
    padding: 0.6rem 1.2rem;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 2px solid #7c3aed;
    color: #c4b5fd;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
[data-testid="stSidebar"] hr {
    border-color: #23263a; margin: 0.75rem 0;
}

/* ---- Don't break words mid-character in narrow containers ---- */
[data-testid="stSidebar"], [data-testid="stSidebar"] *,
[data-testid="stMarkdown"], [data-testid="stMarkdown"] *,
.stDataFrame, .stDataFrame * {
    word-break: normal !important;
    overflow-wrap: break-word !important;
    hyphens: manual !important;
}

/* ---- Rounded cards ---- */
[data-testid="stExpander"], .stDataFrame {
    border-radius: 12px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.35);
    overflow: hidden;
}

/* ---- Chat bubbles ---- */
[data-testid="stChatMessageContent"] {
    border-radius: 14px;
    padding: 0.75rem 1rem;
    margin: 0.25rem 0;
    line-height: 1.55;
}
[data-testid="stChatMessage"][data-testid*="user"] [data-testid="stChatMessageContent"] {
    background: #4c1d95; color: #f3f0ff;
    margin-left: auto; max-width: 75%;
}
[data-testid="stChatMessage"][data-testid*="assistant"] [data-testid="stChatMessageContent"] {
    background: #1e2130; color: #d1d5db; max-width: 85%;
}

/* ---- Inputs ---- */
.stTextInput input, .stTextArea textarea {
    border-radius: 8px;
    border: 1px solid #2e3040;
    background: #161922;
    color: #e2e8f0;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #7c3aed;
    box-shadow: 0 0 0 1px #7c3aed;
}

/* ---- Buttons ---- */
.stButton > button {
    border-radius: 8px;
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white; border: none;
    padding: 0.45rem 1.3rem;
    font-weight: 600;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #5b21b6);
    box-shadow: 0 2px 8px rgba(124,58,237,0.35);
    transform: translateY(-1px);
}

/* ---- HITL container ---- */
[data-testid="stVerticalBlock"] > div:has(> .stSubheader) {
    border: 1px solid #d97706; border-radius: 12px;
}

/* ---- Selectbox ---- */
.stSelectbox > div > div {
    border-radius: 8px;
    border-color: #2e3040;
    background: #161922;
}

/* ---- Welcome hero ---- */
.welcome-hero {
    text-align: center; padding: 3rem 1rem 2rem;
    color: #94a3b8;
}
.welcome-hero h2 { color: #e2e8f0; font-weight: 700; margin-bottom: 0.5rem; }
.welcome-hero p { font-size: 1.05rem; max-width: 520px; margin: 0 auto 1.5rem; }
.cap-grid {
    display: flex; gap: 1rem; justify-content: center;
    flex-wrap: wrap; margin-top: 1rem;
}
.cap-card {
    background: #1a1d2e; border: 1px solid #23263a;
    border-radius: 12px; padding: 1rem 1.25rem;
    width: 200px; text-align: left;
}
.cap-card .icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
.cap-card .title { color: #c4b5fd; font-weight: 600; font-size: 0.9rem; }
.cap-card .desc { color: #64748b; font-size: 0.78rem; margin-top: 0.2rem; }
</style>
"""

_PILL_COLORS = {
    "green": ("background:#16a34a;color:#f0fff4", "●"),
    "slate": ("background:#334155;color:#cbd5e1", "○"),
    "amber": ("background:#d97706;color:#fffbeb", "◉"),
    "red":   ("background:#dc2626;color:#fff5f5", "✕"),
    "purple": ("background:#7c3aed;color:#f5f3ff", "●"),
}


def inject_styles() -> None:
    """Inject the project CSS block into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def status_pill(label: str, color: str = "slate") -> str:
    """Return an HTML span styled as a colored status pill badge."""
    style, icon = _PILL_COLORS.get(color, _PILL_COLORS["slate"])
    return (f'<span style="{style};border-radius:9999px;padding:2px 10px;'
            f'font-size:0.75rem;font-weight:600;display:inline-block;'
            f'margin:2px;">{icon} {label}</span>')
