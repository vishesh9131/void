import { type DragEvent } from 'react';
import { CheckSquare, Bug, Star, AlertCircle, ArrowUp } from 'lucide-react';
import type { KanbanTicket, TicketPriority, TicketType } from '@/types/kanban';
import { useWorkerStore } from '@/stores/workerStore';

interface KanbanCardProps {
  ticket: KanbanTicket;
  onClick: () => void;
}

const PRIORITY_COLORS: Record<TicketPriority, string> = {
  critical: 'bg-red-500/20 text-red-400 border-red-500/30',
  high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  low: 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30',
};

const TYPE_ICONS: Record<TicketType, typeof CheckSquare> = {
  task: CheckSquare,
  bug: Bug,
  feature: Star,
  issue: AlertCircle,
  improvement: ArrowUp,
};

export default function KanbanCard({ ticket, onClick }: KanbanCardProps) {
  const workers = useWorkerStore((s) => s.workers);
  const assignee = ticket.assigneeId ? workers[ticket.assigneeId] : null;
  const TypeIcon = TYPE_ICONS[ticket.type];

  const handleDragStart = (e: DragEvent<HTMLDivElement>) => {
    e.dataTransfer.setData('text/plain', ticket.id);
    e.dataTransfer.effectAllowed = 'move';
  };

  const subtasksDone = ticket.subtasks.filter((s) => s.done).length;
  const subtasksTotal = ticket.subtasks.length;

  return (
    <div
      draggable
      onDragStart={handleDragStart}
      onClick={onClick}
      className="group cursor-pointer rounded-lg border border-aware-border bg-aware-surface p-3 transition-colors hover:border-aware-accent/40 hover:bg-aware-hover active:opacity-80"
    >
      <div className="mb-2 flex items-start gap-2">
        <TypeIcon size={14} className="mt-0.5 shrink-0 text-aware-muted" />
        <span className="flex-1 text-sm font-medium text-aware-text leading-snug">
          {ticket.title}
        </span>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        <span
          className={`rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${PRIORITY_COLORS[ticket.priority]}`}
        >
          {ticket.priority}
        </span>

        {subtasksTotal > 0 && (
          <span className="text-[10px] text-aware-muted">
            {subtasksDone}/{subtasksTotal}
          </span>
        )}

        {assignee && (
          <span className="ml-auto truncate text-[10px] text-aware-worker max-w-[80px]">
            {assignee.name}
          </span>
        )}
      </div>
    </div>
  );
}
