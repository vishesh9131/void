import { FileCode, FolderOpen, FilePlus } from 'lucide-react';

interface WelcomeScreenProps {
  onNewFile?: () => void;
  onOpenFile?: () => void;
  onOpenProject?: () => void;
}

const shortcuts = [
  { keys: 'Ctrl+N', action: 'New File' },
  { keys: 'Ctrl+O', action: 'Open File' },
  { keys: 'Ctrl+S', action: 'Save' },
  { keys: 'Ctrl+Shift+P', action: 'Command Palette' },
  { keys: 'Ctrl+`', action: 'Toggle Terminal' },
  { keys: 'Ctrl+B', action: 'Toggle Sidebar' },
];

export default function WelcomeScreen({ onNewFile, onOpenFile, onOpenProject }: WelcomeScreenProps) {
  return (
    <div className="flex h-full items-center justify-center bg-aware-bg">
      <div className="flex flex-col items-center gap-8 text-center">
        <div className="flex flex-col items-center gap-2">
          <h1 className="text-4xl font-bold tracking-tight text-aware-text">
            Aware
          </h1>
          <p className="text-sm text-aware-muted">
            Canvas-first agentic IDE
          </p>
        </div>

        <div className="flex gap-3">
          <button
            onClick={onNewFile}
            className="flex items-center gap-2 rounded-lg border border-aware-border bg-aware-surface px-4 py-2.5 text-sm text-aware-text hover:bg-aware-hover transition-colors"
          >
            <FilePlus size={16} />
            New File
          </button>
          <button
            onClick={onOpenFile}
            className="flex items-center gap-2 rounded-lg border border-aware-border bg-aware-surface px-4 py-2.5 text-sm text-aware-text hover:bg-aware-hover transition-colors"
          >
            <FileCode size={16} />
            Open File
          </button>
          <button
            onClick={onOpenProject}
            className="flex items-center gap-2 rounded-lg border border-aware-border bg-aware-surface px-4 py-2.5 text-sm text-aware-text hover:bg-aware-hover transition-colors"
          >
            <FolderOpen size={16} />
            Open Project
          </button>
        </div>

        <div className="mt-4 rounded-lg border border-aware-border bg-aware-surface p-4">
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider text-aware-muted">
            Keyboard Shortcuts
          </h3>
          <div className="grid grid-cols-2 gap-x-8 gap-y-2">
            {shortcuts.map((s) => (
              <div key={s.keys} className="flex items-center justify-between gap-6 text-xs">
                <span className="text-aware-muted">{s.action}</span>
                <kbd className="rounded border border-aware-border bg-aware-bg px-1.5 py-0.5 font-mono text-[11px] text-aware-text">
                  {s.keys}
                </kbd>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
