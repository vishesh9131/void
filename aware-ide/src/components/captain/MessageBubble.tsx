import { useState } from 'react';
import { Bot, User, Terminal, ChevronDown, ChevronRight, BrainCircuit } from 'lucide-react';
import type { ChatMessage, ToolCall } from '@/types/agents';

interface MessageBubbleProps {
  message: ChatMessage;
}

function renderContent(content: string): React.ReactNode {
  const parts = content.split(/(```[\s\S]*?```)/g);
  return parts.map((part, i) => {
    if (part.startsWith('```') && part.endsWith('```')) {
      const inner = part.slice(3, -3);
      const newlineIdx = inner.indexOf('\n');
      const code = newlineIdx >= 0 ? inner.slice(newlineIdx + 1) : inner;
      return (
        <pre key={i} className="my-2 rounded bg-aware-bg p-3 text-xs font-mono overflow-x-auto">
          <code>{code}</code>
        </pre>
      );
    }
    return <span key={i} className="whitespace-pre-wrap">{part}</span>;
  });
}

function ToolCallCard({ tc }: { tc: ToolCall }) {
  const [expanded, setExpanded] = useState(false);

  const statusColor =
    tc.status === 'done'
      ? 'text-aware-success'
      : tc.status === 'error'
        ? 'text-aware-error'
        : tc.status === 'running'
          ? 'text-aware-warn'
          : 'text-aware-muted';

  return (
    <div className="my-1.5 rounded border border-aware-border bg-aware-bg p-2 text-xs">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2 text-left"
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <Terminal size={12} className="text-aware-accent shrink-0" />
        <span className="font-medium text-aware-text">{tc.name}</span>
        <span className={`ml-auto text-[10px] ${statusColor}`}>{tc.status}</span>
      </button>
      {expanded && (
        <div className="mt-2 space-y-1.5">
          <pre className="rounded bg-aware-surface p-2 text-[11px] font-mono text-aware-muted overflow-x-auto">
            {JSON.stringify(tc.arguments, null, 2)}
          </pre>
          {tc.result && (
            <pre className="rounded bg-aware-surface p-2 text-[11px] font-mono text-aware-text overflow-x-auto">
              {tc.result}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

function ThinkingSection({ thinking }: { thinking: string }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mb-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-aware-muted hover:text-aware-text transition-colors"
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <BrainCircuit size={12} />
        <span className="italic">Thinking...</span>
      </button>
      {expanded && (
        <p className="mt-1.5 text-xs italic text-aware-muted pl-6 whitespace-pre-wrap">
          {thinking}
        </p>
      )}
    </div>
  );
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';
  const isTool = message.role === 'tool';
  const isSystem = message.role === 'system';

  if (isSystem) {
    return (
      <div className="mx-auto my-2 max-w-lg rounded border border-aware-border bg-aware-bg px-3 py-2 text-center text-xs text-aware-muted">
        {message.content}
      </div>
    );
  }

  if (isTool) {
    return (
      <div className="my-1 ml-10 mr-16">
        <div className="rounded border border-aware-border bg-aware-bg px-3 py-2 text-xs text-aware-muted">
          <div className="flex items-center gap-1.5 mb-1">
            <Terminal size={11} className="text-aware-accent" />
            <span className="font-medium">Tool Result</span>
          </div>
          <pre className="font-mono text-[11px] whitespace-pre-wrap break-words">{message.content}</pre>
        </div>
      </div>
    );
  }

  return (
    <div className={`my-2 flex ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}>
      {!isUser && (
        <div className="mr-2 mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-aware-captain/20">
          <Bot size={14} className="text-aware-captain" />
        </div>
      )}

      <div
        className={`max-w-[75%] rounded-lg px-3.5 py-2.5 text-sm leading-relaxed ${
          isUser
            ? 'bg-aware-accent/20 text-aware-text'
            : 'bg-aware-surface text-aware-text border border-aware-border'
        }`}
      >
        {isAssistant && message.thinking && (
          <ThinkingSection thinking={message.thinking} />
        )}

        {message.content && renderContent(message.content)}

        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="mt-2">
            {message.toolCalls.map((tc) => (
              <ToolCallCard key={tc.id} tc={tc} />
            ))}
          </div>
        )}
      </div>

      {isUser && (
        <div className="ml-2 mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-aware-accent/20">
          <User size={14} className="text-aware-accent" />
        </div>
      )}
    </div>
  );
}
