import { useState } from 'react';
import {
  X,
  CheckSquare,
  Bug,
  Star,
  AlertCircle,
  ArrowUp,
  Plus,
  Link as LinkIcon,
  MessageSquare,
  Send,
} from 'lucide-react';
import type { KanbanTicket, TicketPriority, TicketStatus, TicketType } from '@/types/kanban';
import { useKanbanStore } from '@/stores/kanbanStore';
import { useWorkerStore } from '@/stores/workerStore';

interface TicketDetailProps {
  ticket: KanbanTicket;
  onClose: () => void;
}

const STATUSES: TicketStatus[] = ['backlog', 'todo', 'in_progress', 'review', 'done', 'blocked'];
const PRIORITIES: TicketPriority[] = ['critical', 'high', 'medium', 'low'];
const TYPES: TicketType[] = ['task', 'bug', 'feature', 'issue', 'improvement'];

const TYPE_ICONS: Record<TicketType, typeof CheckSquare> = {
  task: CheckSquare,
  bug: Bug,
  feature: Star,
  issue: AlertCircle,
  improvement: ArrowUp,
};

export default function TicketDetail({ ticket, onClose }: TicketDetailProps) {
  const updateTicket = useKanbanStore((s) => s.updateTicket);
  const moveTicket = useKanbanStore((s) => s.moveTicket);
  const addComment = useKanbanStore((s) => s.addComment);
  const addSubtask = useKanbanStore((s) => s.addSubtask);
  const toggleSubtask = useKanbanStore((s) => s.toggleSubtask);
  const workers = useWorkerStore((s) => s.workers);

  const [commentText, setCommentText] = useState('');
  const [newSubtaskTitle, setNewSubtaskTitle] = useState('');
  const [showSubtaskInput, setShowSubtaskInput] = useState(false);

  const TypeIcon = TYPE_ICONS[ticket.type];
  const workerList = Object.values(workers);

  const handleStatusChange = (status: TicketStatus) => {
    moveTicket(ticket.id, status);
  };

  const handlePriorityChange = (priority: TicketPriority) => {
    updateTicket(ticket.id, { priority });
  };

  const handleTypeChange = (type: TicketType) => {
    updateTicket(ticket.id, { type });
  };

  const handleAssigneeChange = (assigneeId: string) => {
    updateTicket(ticket.id, { assigneeId: assigneeId || undefined });
  };

  const handleAddComment = () => {
    const trimmed = commentText.trim();
    if (!trimmed) return;
    addComment(ticket.id, { author: 'You', content: trimmed });
    setCommentText('');
  };

  const handleAddSubtask = () => {
    const trimmed = newSubtaskTitle.trim();
    if (!trimmed) return;
    addSubtask(ticket.id, { title: trimmed, done: false });
    setNewSubtaskTitle('');
    setShowSubtaskInput(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-xl border border-aware-border bg-aware-panel shadow-2xl">
        {/* Header */}
        <div className="flex items-start justify-between border-b border-aware-border px-6 py-4">
          <div className="flex items-start gap-3">
            <TypeIcon size={18} className="mt-0.5 text-aware-accent" />
            <div>
              <h2 className="text-lg font-semibold text-aware-text">{ticket.title}</h2>
              <p className="mt-0.5 text-xs text-aware-muted">
                Created {new Date(ticket.createdAt).toLocaleDateString()}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded p-1 text-aware-muted hover:bg-aware-hover hover:text-aware-text transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        <div className="px-6 py-4 space-y-5">
          {/* Controls row */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {/* Status */}
            <div>
              <label className="block text-[10px] font-medium uppercase tracking-wider text-aware-muted mb-1">
                Status
              </label>
              <select
                value={ticket.status}
                onChange={(e) => handleStatusChange(e.target.value as TicketStatus)}
                className="w-full rounded border border-aware-border bg-aware-surface px-2 py-1.5 text-xs text-aware-text focus:border-aware-accent focus:outline-none"
              >
                {STATUSES.map((s) => (
                  <option key={s} value={s}>
                    {s.replace('_', ' ')}
                  </option>
                ))}
              </select>
            </div>

            {/* Priority */}
            <div>
              <label className="block text-[10px] font-medium uppercase tracking-wider text-aware-muted mb-1">
                Priority
              </label>
              <select
                value={ticket.priority}
                onChange={(e) => handlePriorityChange(e.target.value as TicketPriority)}
                className="w-full rounded border border-aware-border bg-aware-surface px-2 py-1.5 text-xs text-aware-text focus:border-aware-accent focus:outline-none"
              >
                {PRIORITIES.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>

            {/* Type */}
            <div>
              <label className="block text-[10px] font-medium uppercase tracking-wider text-aware-muted mb-1">
                Type
              </label>
              <select
                value={ticket.type}
                onChange={(e) => handleTypeChange(e.target.value as TicketType)}
                className="w-full rounded border border-aware-border bg-aware-surface px-2 py-1.5 text-xs text-aware-text focus:border-aware-accent focus:outline-none"
              >
                {TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>

            {/* Assignee */}
            <div>
              <label className="block text-[10px] font-medium uppercase tracking-wider text-aware-muted mb-1">
                Assignee
              </label>
              <select
                value={ticket.assigneeId ?? ''}
                onChange={(e) => handleAssigneeChange(e.target.value)}
                className="w-full rounded border border-aware-border bg-aware-surface px-2 py-1.5 text-xs text-aware-text focus:border-aware-accent focus:outline-none"
              >
                <option value="">Unassigned</option>
                {workerList.map((w) => (
                  <option key={w.id} value={w.id}>
                    {w.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Description */}
          <div>
            <h3 className="mb-1.5 text-xs font-medium uppercase tracking-wider text-aware-muted">
              Description
            </h3>
            <p className="rounded bg-aware-surface border border-aware-border p-3 text-sm text-aware-text whitespace-pre-wrap">
              {ticket.description || 'No description provided.'}
            </p>
          </div>

          {/* Subtasks */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-medium uppercase tracking-wider text-aware-muted">
                Subtasks ({ticket.subtasks.filter((s) => s.done).length}/{ticket.subtasks.length})
              </h3>
              <button
                onClick={() => setShowSubtaskInput(true)}
                className="flex items-center gap-1 text-[10px] text-aware-accent hover:text-aware-text transition-colors"
              >
                <Plus size={11} />
                Add Subtask
              </button>
            </div>
            <div className="space-y-1">
              {ticket.subtasks.map((st) => (
                <label
                  key={st.id}
                  className="flex items-center gap-2 rounded px-2 py-1.5 hover:bg-aware-hover cursor-pointer transition-colors"
                >
                  <input
                    type="checkbox"
                    checked={st.done}
                    onChange={() => toggleSubtask(ticket.id, st.id)}
                    className="h-3.5 w-3.5 rounded border-aware-border accent-aware-accent"
                  />
                  <span
                    className={`text-sm ${st.done ? 'line-through text-aware-muted' : 'text-aware-text'}`}
                  >
                    {st.title}
                  </span>
                </label>
              ))}
              {showSubtaskInput && (
                <div className="flex items-center gap-2 px-2">
                  <input
                    autoFocus
                    value={newSubtaskTitle}
                    onChange={(e) => setNewSubtaskTitle(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleAddSubtask();
                      if (e.key === 'Escape') setShowSubtaskInput(false);
                    }}
                    placeholder="Subtask title..."
                    className="flex-1 rounded border border-aware-border bg-aware-surface px-2 py-1 text-xs text-aware-text placeholder:text-aware-muted focus:border-aware-accent focus:outline-none"
                  />
                  <button
                    onClick={handleAddSubtask}
                    className="text-xs text-aware-accent hover:text-aware-text transition-colors"
                  >
                    Add
                  </button>
                </div>
              )}
              {ticket.subtasks.length === 0 && !showSubtaskInput && (
                <p className="px-2 text-xs text-aware-muted">No subtasks</p>
              )}
            </div>
          </div>

          {/* Linked Nodes */}
          {ticket.linkedNodeIds.length > 0 && (
            <div>
              <h3 className="mb-1.5 text-xs font-medium uppercase tracking-wider text-aware-muted">
                Linked Canvas Nodes
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {ticket.linkedNodeIds.map((nodeId) => (
                  <span
                    key={nodeId}
                    className="flex items-center gap-1 rounded border border-aware-border bg-aware-surface px-2 py-1 text-[10px] text-aware-muted"
                  >
                    <LinkIcon size={10} />
                    {nodeId}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Comments */}
          <div>
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-aware-muted">
              <MessageSquare size={12} />
              Comments ({ticket.comments.length})
            </h3>
            <div className="space-y-2 mb-3">
              {ticket.comments.map((c) => (
                <div
                  key={c.id}
                  className="rounded border border-aware-border bg-aware-surface px-3 py-2"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-medium text-aware-text">{c.author}</span>
                    <span className="text-[10px] text-aware-muted">
                      {new Date(c.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <p className="text-xs text-aware-text whitespace-pre-wrap">{c.content}</p>
                </div>
              ))}
              {ticket.comments.length === 0 && (
                <p className="text-xs text-aware-muted">No comments yet</p>
              )}
            </div>

            <div className="flex items-end gap-2">
              <textarea
                value={commentText}
                onChange={(e) => setCommentText(e.target.value)}
                placeholder="Add a comment..."
                rows={2}
                className="flex-1 resize-none rounded border border-aware-border bg-aware-surface px-3 py-2 text-xs text-aware-text placeholder:text-aware-muted focus:border-aware-accent focus:outline-none"
              />
              <button
                onClick={handleAddComment}
                disabled={!commentText.trim()}
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded bg-aware-accent/20 text-aware-accent hover:bg-aware-accent/30 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
              >
                <Send size={14} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
