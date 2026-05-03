import { create } from 'zustand';
import { nanoid } from 'nanoid';
import type { CaptainSession, CaptainMode, ChatMessage } from '@/types/agents';

interface CaptainState {
  sessions: CaptainSession[];
  activeSessionId: string | null;
  isProcessing: boolean;

  createSession: (name?: string, mode?: CaptainMode) => string;
  setActiveSession: (id: string | null) => void;
  deleteSession: (id: string) => void;
  addMessage: (sessionId: string, message: ChatMessage) => void;
  setMode: (sessionId: string, mode: CaptainMode) => void;
  setProcessing: (processing: boolean) => void;
  sendMessage: (content: string) => void;
}

export const useCaptainStore = create<CaptainState>((set, get) => ({
  sessions: [],
  activeSessionId: null,
  isProcessing: false,

  createSession: (name, mode = 'ask') => {
    const id = nanoid();
    const session: CaptainSession = {
      id,
      name: name ?? `Session ${get().sessions.length + 1}`,
      mode,
      messages: [],
      createdAt: Date.now(),
      workerIds: [],
    };
    set((state) => ({
      sessions: [...state.sessions, session],
      activeSessionId: id,
    }));
    return id;
  },

  setActiveSession: (id) => {
    set({ activeSessionId: id });
  },

  deleteSession: (id) => {
    set((state) => ({
      sessions: state.sessions.filter((s) => s.id !== id),
      activeSessionId: state.activeSessionId === id ? null : state.activeSessionId,
    }));
  },

  addMessage: (sessionId, message) => {
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === sessionId ? { ...s, messages: [...s.messages, message] } : s,
      ),
    }));
  },

  setMode: (sessionId, mode) => {
    set((state) => ({
      sessions: state.sessions.map((s) =>
        s.id === sessionId ? { ...s, mode } : s,
      ),
    }));
  },

  setProcessing: (processing) => {
    set({ isProcessing: processing });
  },

  sendMessage: (content) => {
    const { activeSessionId, addMessage, sessions, createSession } = get();
    let sessionId = activeSessionId;

    if (!sessionId || !sessions.find((s) => s.id === sessionId)) {
      sessionId = createSession();
    }

    const message: ChatMessage = {
      id: nanoid(),
      role: 'user',
      content,
      timestamp: Date.now(),
    };

    addMessage(sessionId, message);
  },
}));
