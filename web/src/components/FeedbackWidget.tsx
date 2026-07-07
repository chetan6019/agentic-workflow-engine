// 1–5 star rating + optional comment for a finished run (POST /v1/feedback).
// New capability vs the Streamlit UI, which never wired this endpoint up.

import { useState } from "react";

import * as api from "../api/endpoints";

export function FeedbackWidget({ traceId }: { traceId: string }) {
  const [score, setScore] = useState(0);
  const [comment, setComment] = useState("");
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    try {
      await api.sendFeedback(traceId, score, comment);
      setSent(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  if (sent) return <p className="muted">Thanks for the feedback! ⭐</p>;

  return (
    <div className="feedback">
      <span className="muted">Rate this answer:</span>
      {[1, 2, 3, 4, 5].map((n) => (
        <button key={n} className={`star ${n <= score ? "on" : ""}`}
                onClick={() => setScore(n)} aria-label={`${n} stars`}>
          ★
        </button>
      ))}
      {score > 0 && (
        <>
          <input value={comment} placeholder="Optional comment"
                 onChange={(e) => setComment(e.target.value)} />
          <button onClick={submit}>Send</button>
        </>
      )}
      {error && <span className="error-text">{error}</span>}
    </div>
  );
}
