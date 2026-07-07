// Session history: list, open transcript, rename title.

import { useEffect, useState } from "react";

import * as api from "../api/endpoints";
import type { Message, Session } from "../api/types";
import { Markdown } from "../components/Markdown";

export function History() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [openId, setOpenId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [title, setTitle] = useState("");

  async function refresh() {
    const r = await api.listSessions();
    setSessions(r.sessions);
  }

  useEffect(() => {
    refresh().catch(() => setSessions([]));
  }, []);

  async function open(id: string) {
    setOpenId(id);
    const r = await api.getSession(id);
    setMessages(r.messages);
  }

  async function saveTitle(id: string) {
    await api.renameSession(id, title);
    setRenaming(null);
    await refresh();
  }

  return (
    <div className="history-page">
      <h2>Run history</h2>
      <div className="history-list">
        {sessions.map((s) => (
          <div key={s.id} className={`card session-row ${openId === s.id ? "open" : ""}`}>
            {renaming === s.id ? (
              <>
                <input value={title} onChange={(e) => setTitle(e.target.value)} />
                <button onClick={() => saveTitle(s.id)}>Save</button>
                <button onClick={() => setRenaming(null)}>Cancel</button>
              </>
            ) : (
              <>
                <button className="link" onClick={() => open(s.id)}>{s.title}</button>
                <span className="muted">{s.status} · {s.last_activity}</span>
                <button onClick={() => { setRenaming(s.id); setTitle(s.title); }}>
                  ✏️
                </button>
              </>
            )}
          </div>
        ))}
        {sessions.length === 0 && <p className="muted">No sessions yet.</p>}
      </div>
      {openId && (
        <div className="transcript">
          {messages.map((m, i) => (
            <div key={i} className={`bubble ${m.role}`}>
              <Markdown text={m.content} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
