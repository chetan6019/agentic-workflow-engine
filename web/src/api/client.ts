// Thin fetch wrapper: JSON in/out, Bearer token, uniform errors.
// Mirrors streamlit_app/app.py::_call — the API base is relative ("" ) because
// both the Vite dev server and the prod nginx proxy make /v1 same-origin.

// The JWT lives in MEMORY only (no localStorage) — deliberate: refreshing or
// reloading the page drops the token, which disconnects every open connection
// (SSE stream, polls) and returns the user to the login page for a clean slate.
// Side benefit: the token is out of reach of anything that reads localStorage.
let accessToken: string | null = null;

// One-time cleanup: an earlier build stored the JWT in localStorage under this
// key. Purge it so stale sessions can't survive a reload via the old entry.
localStorage.removeItem("awe_token");

export function getToken(): string | null {
  return accessToken;
}

export function setToken(token: string): void {
  accessToken = token;
}

export function clearToken(): void {
  accessToken = null;
}

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

interface RequestOptions {
  body?: unknown;
  headers?: Record<string, string>;
  timeoutMs?: number;
}

export async function request<T>(method: string, path: string,
                                 opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { ...opts.headers };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";

  const resp = await fetch(path, {
    method,
    headers,
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
    signal: opts.timeoutMs ? AbortSignal.timeout(opts.timeoutMs) : undefined,
  });

  if (resp.status === 401 && !path.startsWith("/v1/auth/")) {
    // Session expired (24h JWT, no refresh endpoint): log out and start over.
    clearToken();
    window.location.assign("/login?expired=1");
    throw new ApiError(401, "session_expired");
  }
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const data = await resp.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data);
    } catch {
      /* non-JSON error body — keep statusText */
    }
    throw new ApiError(resp.status, detail);
  }
  return (await resp.json()) as T;
}
