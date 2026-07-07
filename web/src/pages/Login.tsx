import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";

export function Login() {
  const { login, register } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await (mode === "login" ? login(username, password)
                              : register(username, password));
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-page">
      <h1>Agentic Workflow Engine</h1>
      {params.get("expired") && (
        <p className="error-text">Your session expired — please sign in again.</p>
      )}
      <div className="tab-row">
        <button className={mode === "login" ? "active" : ""}
                onClick={() => setMode("login")}>Sign in</button>
        <button className={mode === "register" ? "active" : ""}
                onClick={() => setMode("register")}>Register</button>
      </div>
      <form onSubmit={submit} className="card">
        <input value={username} placeholder="Username (min 3 chars)"
               autoComplete="username"
               onChange={(e) => setUsername(e.target.value)} />
        <input type="password" value={password} placeholder="Password (min 8 chars)"
               autoComplete={mode === "login" ? "current-password" : "new-password"}
               onChange={(e) => setPassword(e.target.value)} />
        <button type="submit" disabled={busy || username.length < 3 || password.length < 8}>
          {mode === "login" ? "Sign in" : "Create account"}
        </button>
        {error && <p className="error-text">{error}</p>}
      </form>
    </div>
  );
}
