// Auth state: JWT held in memory only (see api/client.ts — a page reload
// deliberately disconnects everything and returns to login), login/register/logout.

import { createContext, useCallback, useContext, useState } from "react";
import type { ReactNode } from "react";

import { clearToken, getToken, setToken } from "../api/client";
import * as api from "../api/endpoints";

interface AuthValue {
  authenticated: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [authenticated, setAuthenticated] = useState<boolean>(() => !!getToken());

  const login = useCallback(async (username: string, password: string) => {
    const resp = await api.login(username, password);
    setToken(resp.access_token);
    setAuthenticated(true);
  }, []);

  const register = useCallback(async (username: string, password: string) => {
    const resp = await api.register(username, password);
    setToken(resp.access_token);
    setAuthenticated(true);
  }, []);

  const logout = useCallback(() => {
    clearToken();
    setAuthenticated(false);
  }, []);

  return (
    <AuthContext.Provider value={{ authenticated, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside <AuthProvider>");
  return value;
}
