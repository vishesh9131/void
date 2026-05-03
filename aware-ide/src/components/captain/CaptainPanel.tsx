import { useEffect, useRef } from 'react';
import { Plus, MessageSquare, Trash2 } from 'lucide-react';
import { useCaptainStore } from '@/stores/captainStore';
import type { CaptainMode } from '@/types/agents';
import MessageBubble from './MessageBubble';
import ChatInput from './ChatInput';
import { useCaptainRunner } from './CaptainRunner';

export default function CaptainPanel() {
  const sessions = useCaptainStore((s) => s.sessions);
  const activeSessionId = useCaptainStore((s) => s.activeSessionId);
  const createSession = useCaptainStore((s) => s.createSession);
  const setActiveSession = useCaptainStore((s) => s.setActiveSession);
  const deleteSession = useCaptainStore((s) => s.deleteSession);
  const sendMessage = useCaptainStore((s) => s.sendMessage);
  const setMode = useCaptainStore((s) => s.setMode);

  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { run, stop } = useCaptainRunner();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activeSession?.messages.length]);

  const handleSend = (content: string) => {
    sendMessage(content);
    setTimeout(() => run(), 0);
  };

  const handleModeToggle = (mode: CaptainMode) => {
    if (activeSessionId) {
      setMode(activeSessionId, mode);
    }
  };

  return (
    <div className="flex h-full bg-aware-bg">
      {/* Sidebar */}
      <div className="flex w-[200px] shrink-0 flex-col border-r border-aware-border bg-aware-panel">
        <div className="flex items-center justify-between border-b border-aware-border px-3 py-2.5">
          <span className="text-xs font-semibold uppercase tracking-wider text-aware-captain">
            Captain
          </span>
          <button
            onClick={() => createSession()}
            className="rounded p-1 text-aware-muted hover:bg-aware-hover hover:text-aware-text transition-colors"
            title="New Session"
          >
            <Plus size={14} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {sessions.length === 0 && (
            <p className="px-3 py-4 text-xs text-aware-muted">No sessions yet</p>
          )}
          {sessions.map((session) => (
            <div
              key={session.id}
              className={`group flex cursor-pointer items-center gap-2 border-b border-aware-border px-3 py-2.5 transition-colors ${
                session.id === activeSessionId
                  ? 'bg-aware-surface text-aware-text'
                  : 'text-aware-muted hover:bg-aware-hover hover:text-aware-text'
              }`}
              onClick={() => setActiveSession(session.id)}
            >
              <MessageSquare size={13} className="shrink-0" />
              <span className="flex-1 truncate text-xs">{session.name}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteSession(session.id);
                }}
                className="hidden rounded p-0.5 text-aware-muted hover:text-aware-error group-hover:block"
                title="Delete session"
              >
                <Trash2 size={11} />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex flex-1 flex-col">
        {activeSession ? (
          <>
            {/* Top bar */}
            <div className="flex items-center justify-between border-b border-aware-border bg-aware-panel px-4 py-2">
              <span className="text-sm font-medium text-aware-text truncate">
                {activeSession.name}
              </span>

              {/* Mode toggle */}
              <div className="flex rounded-lg border border-aware-border bg-aware-bg p-0.5">
                <button
                  onClick={() => handleModeToggle('ask')}
                  className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                    activeSession.mode === 'ask'
                      ? 'bg-aware-captain text-white'
                      : 'text-aware-muted hover:text-aware-text'
                  }`}
                >
                  Ask
                </button>
                <button
                  onClick={() => handleModeToggle('build')}
                  className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                    activeSession.mode === 'build'
                      ? 'bg-aware-captain text-white'
                      : 'text-aware-muted hover:text-aware-text'
                  }`}
                >
                  Build
                </button>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-4">
              {activeSession.messages.length === 0 && (
                <div className="flex h-full items-center justify-center">
                  <p className="text-sm text-aware-muted">
                    Start a conversation with Captain
                  </p>
                </div>
              )}
              {activeSession.messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <ChatInput onSend={handleSend} onStop={stop} />
          </>
        ) : (
          <div className="flex flex-1 items-center justify-center">
            <div className="text-center">
              <p className="text-sm text-aware-muted mb-3">
                Select a session or create a new one
              </p>
              <button
                onClick={() => createSession()}
                className="rounded-lg bg-aware-captain/20 px-4 py-2 text-sm font-medium text-aware-captain hover:bg-aware-captain/30 transition-colors"
              >
                New Session
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
