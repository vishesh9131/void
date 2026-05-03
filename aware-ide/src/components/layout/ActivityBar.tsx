import { LayoutGrid, BrainCircuit, Code, Columns3, Settings } from 'lucide-react';

export type ViewId = 'canvas' | 'captain' | 'editor' | 'kanban' | 'settings';

interface ActivityBarProps {
  activeView: ViewId;
  onViewChange: (view: ViewId) => void;
}

const NAV_ITEMS: { id: ViewId; icon: typeof LayoutGrid; label: string }[] = [
  { id: 'canvas', icon: LayoutGrid, label: 'Canvas' },
  { id: 'captain', icon: BrainCircuit, label: 'Captain' },
  { id: 'editor', icon: Code, label: 'Editor' },
  { id: 'kanban', icon: Columns3, label: 'Kanban' },
  { id: 'settings', icon: Settings, label: 'Settings' },
];

export default function ActivityBar({ activeView, onViewChange }: ActivityBarProps) {
  return (
    <div className="flex h-full w-12 shrink-0 flex-col items-center border-r border-aware-border bg-aware-surface">
      {/* Logo */}
      <div className="flex h-12 w-full items-center justify-center border-b border-aware-border">
        <div className="flex h-7 w-7 items-center justify-center rounded-full bg-aware-accent/20">
          <span className="text-xs font-bold text-aware-accent">A</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col items-center gap-1 pt-2">
        {NAV_ITEMS.map(({ id, icon: Icon, label }) => {
          const isActive = activeView === id;
          return (
            <button
              key={id}
              onClick={() => onViewChange(id)}
              className={`relative flex h-10 w-10 items-center justify-center rounded-md transition-colors ${
                isActive
                  ? 'text-aware-text'
                  : 'text-aware-muted hover:text-aware-text hover:bg-aware-hover'
              }`}
              title={label}
            >
              {isActive && (
                <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-r bg-aware-accent" />
              )}
              <Icon size={20} strokeWidth={isActive ? 2 : 1.5} />
            </button>
          );
        })}
      </nav>
    </div>
  );
}
