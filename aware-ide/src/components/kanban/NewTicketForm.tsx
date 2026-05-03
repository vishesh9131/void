import { useState } from 'react';
import { X } from 'lucide-react';
import { useKanbanStore } from '@/stores/kanbanStore';
import { useWorkerStore } from '@/stores/workerStore';
import type { TicketType, TicketPriority } from '@/types/kanban';

interface NewTicketFormProps {
  onClose: () => void;
}

const TYPES: TicketType[] = ['task', 'bug', 'feature', 'issue', 'improvement'];
const PRIORITIES: TicketPriority[] = ['critical', 'high', 'medium', 'low'];

export default function NewTicketForm({ onClose }: NewTicketFormProps) {
  const addTicket = useKanbanStore((s) => s.addTicket);
  const workers = useWorkerStore((s) => s.workers);
  const workerList = Object.values(workers);

  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [type, setType] = useState<TicketType>('task');
  const [priority, setPriority] = useState<TicketPriority>('medium');
  const [assigneeId, setAssigneeId] = useState('');
  const [tagsInput, setTagsInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedTitle = title.trim();
    if (!trimmedTitle) return;

    const tags = tagsInput
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean);

    addTicket({
      title: trimmedTitle,
      description: description.trim(),
      type,
      priority,
      status: 'backlog',
      assigneeId: assigneeId || undefined,
      reporterId: 'user',
      tags,
      subtasks: [],
      comments: [],
      linkedNodeIds: [],
    });

    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-md rounded-xl border border-aware-border bg-aware-panel shadow-2xl">
        <div className="flex items-center justify-between border-b border-aware-border px-5 py-3">
          <h2 className="text-sm font-semibold text-aware-text">New Ticket</h2>
          <button
            onClick={onClose}
            className="rounded p-1 text-aware-muted hover:bg-aware-hover hover:text-aware-text transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="px-5 py-4 space-y-4">
          {/* Title */}
          <div>
            <label className="block text-[10px] font-medium uppercase tracking-wider text-aware-muted mb-1">
              Title *
            </label>
            <input
              autoFocus
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Ticket title..."
              className="w-full rounded border border-aware-border bg-aware-surface px-3 py-2 text-sm text-aware-text placeholder:text-aware-muted focus:border-aware-accent focus:outline-none"
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-[10px] font-medium uppercase tracking-wider text-aware-muted mb-1">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe the task..."
              rows={3}
              className="w-full resize-none rounded border border-aware-border bg-aware-surface px-3 py-2 text-sm text-aware-text placeholder:text-aware-muted focus:border-aware-accent focus:outline-none"
            />
          </div>

          {/* Type + Priority row */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-[10px] font-medium uppercase tracking-wider text-aware-muted mb-1">
                Type
              </label>
              <select
                value={type}
                onChange={(e) => setType(e.target.value as TicketType)}
                className="w-full rounded border border-aware-border bg-aware-surface px-2 py-2 text-sm text-aware-text focus:border-aware-accent focus:outline-none"
              >
                {TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-[10px] font-medium uppercase tracking-wider text-aware-muted mb-1">
                Priority
              </label>
              <select
                value={priority}
                onChange={(e) => setPriority(e.target.value as TicketPriority)}
                className="w-full rounded border border-aware-border bg-aware-surface px-2 py-2 text-sm text-aware-text focus:border-aware-accent focus:outline-none"
              >
                {PRIORITIES.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Assignee */}
          <div>
            <label className="block text-[10px] font-medium uppercase tracking-wider text-aware-muted mb-1">
              Assignee
            </label>
            <select
              value={assigneeId}
              onChange={(e) => setAssigneeId(e.target.value)}
              className="w-full rounded border border-aware-border bg-aware-surface px-2 py-2 text-sm text-aware-text focus:border-aware-accent focus:outline-none"
            >
              <option value="">Unassigned</option>
              {workerList.map((w) => (
                <option key={w.id} value={w.id}>
                  {w.name}
                </option>
              ))}
            </select>
          </div>

          {/* Tags */}
          <div>
            <label className="block text-[10px] font-medium uppercase tracking-wider text-aware-muted mb-1">
              Tags (comma-separated)
            </label>
            <input
              value={tagsInput}
              onChange={(e) => setTagsInput(e.target.value)}
              placeholder="frontend, api, urgent"
              className="w-full rounded border border-aware-border bg-aware-surface px-3 py-2 text-sm text-aware-text placeholder:text-aware-muted focus:border-aware-accent focus:outline-none"
            />
          </div>

          {/* Actions */}
          <div className="flex items-center justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg px-4 py-2 text-sm text-aware-muted hover:bg-aware-hover hover:text-aware-text transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!title.trim()}
              className="rounded-lg bg-aware-accent px-4 py-2 text-sm font-medium text-white hover:bg-aware-accent-dim transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Create Ticket
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
