import { Bot, Cloud, Server, CheckCircle, AlertCircle, Circle } from 'lucide-react';
import type { LLMProvider } from '@/types/llm';
import type { ReactNode } from 'react';

type ConnectionStatus = 'connected' | 'error' | 'unconfigured';

interface ProviderCardProps {
  provider: LLMProvider;
  isActive: boolean;
  status: ConnectionStatus;
  onClick: () => void;
  children: ReactNode;
}

const providerMeta: Record<LLMProvider, { label: string; icon: ReactNode }> = {
  anthropic: {
    label: 'Anthropic',
    icon: <Bot size={18} />,
  },
  openai: {
    label: 'OpenAI',
    icon: <Cloud size={18} />,
  },
  vllm: {
    label: 'vLLM',
    icon: <Server size={18} />,
  },
};

const statusMeta: Record<ConnectionStatus, { icon: ReactNode; label: string; color: string }> = {
  connected: {
    icon: <CheckCircle size={12} />,
    label: 'Connected',
    color: 'text-aware-success',
  },
  error: {
    icon: <AlertCircle size={12} />,
    label: 'Error',
    color: 'text-aware-error',
  },
  unconfigured: {
    icon: <Circle size={12} />,
    label: 'Not configured',
    color: 'text-aware-muted',
  },
};

export default function ProviderCard({
  provider,
  isActive,
  status,
  onClick,
  children,
}: ProviderCardProps) {
  const meta = providerMeta[provider];
  const sMeta = statusMeta[status];

  return (
    <div
      className={`rounded-xl border transition-colors ${
        isActive
          ? 'border-aware-accent bg-aware-surface'
          : 'border-aware-border bg-aware-panel hover:border-aware-hover cursor-pointer'
      }`}
    >
      <div
        className="flex items-center justify-between px-4 py-3"
        onClick={!isActive ? onClick : undefined}
      >
        <div className="flex items-center gap-2.5">
          <span className={isActive ? 'text-aware-accent' : 'text-aware-muted'}>
            {meta.icon}
          </span>
          <span className="text-sm font-medium text-aware-text">{meta.label}</span>
        </div>
        <div className={`flex items-center gap-1.5 text-xs ${sMeta.color}`}>
          {sMeta.icon}
          <span>{sMeta.label}</span>
        </div>
      </div>

      {isActive && (
        <div className="border-t border-aware-border px-4 py-4">
          {children}
        </div>
      )}
    </div>
  );
}
