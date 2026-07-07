// Live progress for one run: SSE first (GET /v1/invoke/stream/{id}), with a
// polling fallback (/phase + /result — the same flow Streamlit uses) if the
// stream transport fails. EventSource can't send headers, so the JWT rides as
// ?access_token= (the endpoint validates it with the same decoder).

import { useEffect, useRef, useState } from "react";

import { getToken } from "../api/client";
import * as api from "../api/endpoints";
import type { RunResult } from "../api/types";

const POLL_MS = 1500;

export interface RunStream {
  phase: string;
  result: RunResult | null;
  failed: boolean;
}

export function useRunStream(traceId: string | null): RunStream {
  const [phase, setPhase] = useState<string>("🚀 starting");
  // The result is stored TAGGED with the trace it belongs to, and only returned
  // while that trace is still the active one. Without the tag, a re-render
  // between "new traceId set" and "state reset" briefly exposes the PREVIOUS
  // run's result to the caller, which showed the old answer under a new query.
  const [tagged, setTagged] =
    useState<{ traceId: string; result: RunResult } | null>(null);
  const [failed, setFailed] = useState(false);
  // The result value inside the EventSource callbacks would be stale; track via ref.
  const doneRef = useRef(false);

  useEffect(() => {
    if (!traceId) return;
    doneRef.current = false;
    setPhase("🚀 starting");
    setFailed(false);

    let cancelled = false;
    let poller: number | undefined;

    const finish = (r: RunResult) => {
      if (cancelled || doneRef.current) return;
      doneRef.current = true;
      setTagged({ traceId, result: r });
      if (r.status === "failed") setFailed(true);
    };

    const startPolling = () => {
      poller = window.setInterval(async () => {
        try {
          const p = await api.phase(traceId);
          if (cancelled) return;
          setPhase(p.phase);
          if (p.done) {
            window.clearInterval(poller);
            finish(await api.result(traceId));
          }
        } catch {
          /* transient poll error — next tick retries */
        }
      }, POLL_MS);
    };

    const token = getToken() ?? "";
    const es = new EventSource(
      `/v1/invoke/stream/${traceId}?access_token=${encodeURIComponent(token)}`);

    es.addEventListener("phase", (e) => {
      setPhase(JSON.parse((e as MessageEvent).data).phase);
    });
    es.addEventListener("result", (e) => {
      es.close(); // close BEFORE state updates so EventSource never auto-reconnects
      finish(JSON.parse((e as MessageEvent).data) as RunResult);
    });
    es.addEventListener("error", (e) => {
      // Server-sent error event (stream_timeout / not_found) — terminal.
      es.close();
      const data = (e as MessageEvent).data;
      if (data) {
        setFailed(true);
        doneRef.current = true;
      } else if (!doneRef.current && !cancelled) {
        // Transport-level failure (proxy restart, network): degrade to polling.
        startPolling();
      }
    });

    return () => {
      cancelled = true;
      es.close();
      if (poller) window.clearInterval(poller);
    };
  }, [traceId]);

  return {
    phase,
    result: tagged && tagged.traceId === traceId ? tagged.result : null,
    failed,
  };
}
