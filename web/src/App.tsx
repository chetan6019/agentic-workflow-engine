import { Navigate, NavLink, Outlet, Route, Routes, useLocation } from "react-router-dom";

import { useAuth } from "./auth/AuthContext";
import { ChatProvider } from "./chat/ChatStore";
import { Approvals } from "./pages/Approvals";
import { Chat } from "./pages/Chat";
import { History } from "./pages/History";
import { Integrations } from "./pages/Integrations";
import { Login } from "./pages/Login";

const NAV = [
  { to: "/", label: "Chat", icon: "💬" },
  { to: "/history", label: "History", icon: "🕘" },
  { to: "/approvals", label: "Approvals", icon: "🛑" },
  { to: "/integrations", label: "Integrations", icon: "🔌" },
];

function Shell() {
  const { authenticated, logout } = useAuth();
  const location = useLocation();
  if (!authenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  // ChatProvider sits here (not around a single route) so the transcript
  // survives switching pages; it unmounts — and clears — on logout.
  return (
    <ChatProvider>
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-icon">⚙️</span>
          <span>
            <strong>Workflow</strong>
            <br />
            <small>Engine</small>
          </span>
        </div>
        <nav>
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.to === "/"}
                     className={({ isActive }) => (isActive ? "active" : "")}>
              <span className="nav-icon">{item.icon}</span> {item.label}
            </NavLink>
          ))}
        </nav>
        <button className="signout" onClick={logout}>⏻ Sign out</button>
      </aside>
      <main>
        <Outlet />
      </main>
    </div>
    </ChatProvider>
  );
}

export function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route element={<Shell />}>
        <Route path="/" element={<Chat />} />
        <Route path="/history" element={<History />} />
        <Route path="/approvals" element={<Approvals />} />
        <Route path="/integrations" element={<Integrations />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
