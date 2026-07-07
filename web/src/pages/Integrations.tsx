// Integrations: provider cards with status, manual token-paste connect form,
// disconnect. Legacy gmail/calendar rows fold into "google" (port of
// streamlit_app/app.py::_canon_provider) — disconnect removes each raw row.

import { useEffect, useState } from "react";

import * as api from "../api/endpoints";
import type { Integration } from "../api/types";

const PANEL_PROVIDERS = ["google", "github", "reddit"] as const;
const LEGACY_GOOGLE = new Set(["gmail", "calendar"]);

function canonProvider(provider: string): string {
  return LEGACY_GOOGLE.has(provider) ? "google" : provider;
}

export function Integrations() {
  const [rows, setRows] = useState<Integration[]>([]);
  const [connecting, setConnecting] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const [refreshToken, setRefreshToken] = useState("");
  const [expiresIn, setExpiresIn] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    const r = await api.listIntegrations();
    setRows(r.integrations);
  }

  useEffect(() => {
    refresh().catch(() => setRows([]));
  }, []);

  // Best status per canonical provider (a valid row wins over an expiring one).
  const statusFor = (provider: string): string | null => {
    const matches = rows.filter((r) => canonProvider(r.provider) === provider);
    if (matches.length === 0) return null;
    const order = { valid: 0, expiring: 1, revoked: 2 } as Record<string, number>;
    matches.sort((a, b) => (order[a.status] ?? 3) - (order[b.status] ?? 3));
    return matches[0].status;
  };

  async function connect(provider: string) {
    setError(null);
    try {
      await api.saveIntegrationToken(provider, token,
                                     refreshToken || undefined,
                                     expiresIn ? Number(expiresIn) : undefined);
      setConnecting(null);
      setToken("");
      setRefreshToken("");
      setExpiresIn("");
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function disconnect(provider: string) {
    // Delete every raw row that folds into this canonical provider.
    const raws = rows.filter((r) => canonProvider(r.provider) === provider);
    for (const r of raws) {
      await api.deleteIntegration(r.provider).catch(() => undefined);
    }
    await refresh();
  }

  return (
    <div className="integrations-page">
      <h2>Integrations</h2>
      <p className="muted">
        Paste an access token obtained from the provider (e.g. Google OAuth
        Playground). A redirect-based connect flow is a planned upgrade.
      </p>
      {PANEL_PROVIDERS.map((provider) => {
        const status = statusFor(provider);
        return (
          <div key={provider} className="card session-row">
            <strong>{provider}</strong>
            <span className={`pill pill-${status === "valid" ? "green" : status ? "amber" : "red"}`}>
              {status ?? "not connected"}
            </span>
            {status ? (
              <button onClick={() => disconnect(provider)}>Disconnect</button>
            ) : (
              <button onClick={() => setConnecting(provider)}>Connect</button>
            )}
          </div>
        );
      })}
      {connecting && (
        <div className="card">
          <h3>Connect {connecting}</h3>
          <input value={token} placeholder="Access token"
                 onChange={(e) => setToken(e.target.value)} />
          <input value={refreshToken} placeholder="Refresh token (optional)"
                 onChange={(e) => setRefreshToken(e.target.value)} />
          <input value={expiresIn} placeholder="Expires in seconds (optional)"
                 onChange={(e) => setExpiresIn(e.target.value)} />
          <div className="button-row">
            <button disabled={!token} onClick={() => connect(connecting)}>Save</button>
            <button onClick={() => setConnecting(null)}>Cancel</button>
          </div>
          {error && <p className="error-text">{error}</p>}
        </div>
      )}
    </div>
  );
}
