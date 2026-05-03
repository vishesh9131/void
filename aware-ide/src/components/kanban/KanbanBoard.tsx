import { useState, type DragEvent } from 'react';
import { Plus, Filter } from 'lucide-react';
import { useKanbanStore } from '@/stores/kanbanStore';
import { useWorkerStore } from '@/stores/workerStore';
import type { TicketStatus } from '@/types/kanban';
import KanbanCard from './KanbanCard';
import TicketDetail from './TicketDetail';
import NewTicketForm from './NewTicketForm';

const STATUS_COLORS: Record<TicketStatus, string> = {
  backlog: 'text-zinc-400',
  todo: 'text-blue-400',
  in_progress: 'text-yellow-400',
  review: 'text-purple-400',
  done: 'text-green-400',
  blocked: 'text-red-400',
};

export default function KanbanBoard() {
  const columns = useKanbanStore((s) => s.columns);
  const tickets = useKanbanStore((s) => s.tickets);
  const moveTicket = useKanbanStore((s) => s.moveTicket);
  const workers = useWorkerStore((s) => s.workers);

  const [selectedTicketId, setSelectedTicketId] = useState<string | null>(null);
  const [showNewTicket, setShowNewTicket] = useState(false);
  const [filterAssignee, setFilterAssignee] = useState<string>('');
  const [dragOverCol, setDragOverCol] = useState<string | null>(null);

  const workerList = Object.values(workers);
  const selectedTicket = selectedTicketId ? tickets[selectedTicketId] : null;

  const handleDragOver = (e: DragEvent<HTMLDivElement>, colId: string) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    setDragOverCol(colId);
  };

  const handleDragLeave = () => {
    setDragOverCol(null);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>, targetStatus: TicketStatus) => {
    e.preventDefault();
    setDragOverCol(null);
    const ticketId = e.dataTransfer.getData('text/plain');
    if (ticketId) {
      moveTicket(ticketId, targetStatus);
    }
  };

  return (
    <div className="flex h-full flex-col bg-aware-bg">
      {/* Top bar */}
      <div className="flex items-center justify-between border-b border-aware-border bg-aware-panel px-4 py-2.5">
        <h2 className="text-sm font-semibold text-aware-text">Kanban Board</h2>
        <div className="flex items-center gap-3">
          {/* Filter */}
          <div className="flex items-center gap-1.5">
            <Filter size={13} className="text-aware-muted" />
            <select
              value={filterAssignee}
              onChange={(e) => setFilterAssignee(e.target.value)}
              className="rounded border border-aware-border bg-aware-surface px-2 py-1 text-xs text-aware-text focus:border-aware-accent focus:outline-none"
            >
              <option value="">All assignees</option>
              <option value="__unassigned__">Unassigned</option>
              {workerList.map((w) => (
                <option key={w.id} value={w.id}>
                  {w.name}
                </option>
              ))}
            </select>
          </div>

          <button
            onClick={() => setShowNewTicket(true)}
            className="flex items-center gap-1.5 rounded-lg bg-aware-accent/20 px-3 py-1.5 text-xs font-medium text-aware-accent hover:bg-aware-accent/30 transition-colors"
          >
            <Plus size={13} />
            New Ticket
          </button>
        </div>
      </div>

      {/* Board */}
      <div className="flex flex-1 gap-3 overflow-x-auto p-4">
        {columns.map((col) => {
          const colTickets = col.ticketIds
            .map((id) => tickets[id])
            .filter(Boolean)
            .filter((t) => {
              if (!filterAssignee) return true;
              if (filterAssignee === '__unassigned__') return !t.assigneeId;
              return t.assigneeId === filterAssignee;
            });

          return (
            <div
              key={col.id}
              className={`flex w-[260px] shrink-0 flex-col rounded-lg border bg-aware-panel ${
                dragOverCol === col.id
                  ? 'border-aware-accent/50 bg-aware-accent/5'
                  : 'border-aware-border'
              }`}
              onDragOver={(e) => handleDragOver(e, col.id)}
              onDragLeave={handleDragLeave}
              onDrop={(e) => handleDrop(e, col.status)}
            >
              {/* Column header */}
              <div className="flex items-center justify-between border-b border-aware-border px-3 py-2.5">
                <span className={`text-xs font-semibold uppercase tracking-wider ${STATUS_COLORS[col.status]}`}>
                  {col.title}
                </span>
                <span className="rounded-full bg-aware-bg px-1.5 py-0.5 text-[10px] text-aware-muted">
                  {colTickets.length}
                </span>
              </div>

              {/* Cards */}
              <div className="flex-1 space-y-2 overflow-y-auto p-2">
                {colTickets.map((ticket) => (
                  <KanbanCard
                    key={ticket.id}
                    ticket={ticket}
                    onClick={() => setSelectedTicketId(ticket.id)}
                  />
                ))}
                {colTickets.length === 0 && (
                  <p className="py-6 text-center text-[10px] text-aware-muted">
                    No tickets
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Modals */}
      {selectedTicket && (
        <TicketDetail
          ticket={selectedTicket}
          onClose={() => setSelectedTicketId(null)}
        />
      )}
      {showNewTicket && (
        <NewTicketForm onClose={() => setShowNewTicket(false)} />
      )}
    </div>
  );
}
