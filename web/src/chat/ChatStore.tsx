// Chat state lifted OUT of the Chat route component so switching to another
// page (History/Approvals/Integrations) and back doesn't flush the transcript.
// The provider mounts inside the authenticated Shell: state survives route
// changes, and is dropped on logout or a full page reload (deliberate — a
// reload is a clean slate, see api/client.ts token note).

import { createContext, useContext, useState } from "react";
import type { Dispatch, ReactNode, SetStateAction } from "react";

import type { AgentState } from "../api/types";

// One chat entry: a user prompt (text) or an assistant answer (full run state).
export interface Entry {
  role: "user" | "assistant";
  text?: string;
  state?: AgentState | null;
}

interface ChatStore {
  entries: Entry[];
  setEntries: Dispatch<SetStateAction<Entry[]>>;
  traceId: string | null;
  setTraceId: Dispatch<SetStateAction<string | null>>;
  pendingApproval: AgentState | null;
  setPendingApproval: Dispatch<SetStateAction<AgentState | null>>;
  requireApproval: boolean;
  setRequireApproval: Dispatch<SetStateAction<boolean>>;
}

const Ctx = createContext<ChatStore | null>(null);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [traceId, setTraceId] = useState<string | null>(null);
  const [pendingApproval, setPendingApproval] = useState<AgentState | null>(null);
  const [requireApproval, setRequireApproval] = useState(false);

  return (
    <Ctx.Provider value={{ entries, setEntries, traceId, setTraceId,
                           pendingApproval, setPendingApproval,
                           requireApproval, setRequireApproval }}>
      {children}
    </Ctx.Provider>
  );
}

export function useChatStore(): ChatStore {
  const value = useContext(Ctx);
  if (!value) throw new Error("useChatStore must be used inside <ChatProvider>");
  return value;
}
