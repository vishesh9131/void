import { Circle } from 'lucide-react';
import { useSettingsStore } from '@/stores/settingsStore';
import { useWorkerStore } from '@/stores/workerStore';
import type { ViewId } from './ActivityBar';

interface StatusBarProps {
  onNavigate: (view: ViewId) => void;
}

export default function StatusBar({ onNavigate }: StatusBarProps) {
  const llmConfig = useSettingsStore((s) => s.llmConfig);
  const projectPath = useSettingsStore((s) => s.projectPath);
  const workers = useWorkerStore((s) => s.workers);

  const workerList = Object.values(workers);
  const activeCount = workerList.filter((w) => w.status === 'working').length;
  const totalCount = workerList.length;

  const providerLabel = llmConfig.provider.toUpperCase();
  const modelLabel = llmConfig.model || 'no model';
  const isConnected = llmConfig.provider === 'vllm'
    ? Boolean(llmConfig.baseUrl)
    : Boolean(llmConfig.apiKey);

  return (
    <div className="flex h-6 w-full shrink-0 items-center justify-between border-t border-aware-border bg-aware-surface px-3 text-[11px] text-aware-muted">
      <div className="flex items-center gap-4">
        {/* Provider + Model */}
        <button
          onClick={() => onNavigate('settings')}
          className="flex items-center gap-1.5 hover:text-aware-text transition-colors"
          title="Open settings"
        >
          <span className="font-medium">{providerLabel}</span>
          <span className="text-aware-muted/70">{modelLabel}</span>
        </button>

        {/* Project path */}
        {projectPath && (
          <span className="hidden sm:inline truncate max-w-[200px]" title={projectPath}>
            {projectPath}
          </span>
        )}
      </div>

      <div className="flex items-center gap-4">
        {/* Workers */}
        <span className="flex items-center gap-1.5">
          {totalCount > 0 ? (
            <>
              <span>{activeCount} active</span>
              <span className="text-aware-muted/50">/</span>
              <span>{totalCount} workers</span>
            </>
          ) : (
            <span>No workers</span>
          )}
        </span>

        {/* Connection status */}
        <span className="flex items-center gap-1">
          <Circle
            size={7}
            fill={isConnected ? '#22c55e' : '#71717a'}
            strokeWidth={0}
          />
          <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
        </span>
      </div>
    </div>
  );
}
