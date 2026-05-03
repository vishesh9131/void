import { create } from 'zustand';
import { nanoid } from 'nanoid';
import type {
  KanbanTicket,
  KanbanColumn,
  TicketStatus,
  Subtask,
  TicketComment,
} from '@/types/kanban';

const DEFAULT_COLUMNS: KanbanColumn[] = [
  { id: 'col-backlog', title: 'Backlog', status: 'backlog', ticketIds: [] },
  { id: 'col-todo', title: 'To Do', status: 'todo', ticketIds: [] },
  { id: 'col-in_progress', title: 'In Progress', status: 'in_progress', ticketIds: [] },
  { id: 'col-review', title: 'Review', status: 'review', ticketIds: [] },
  { id: 'col-done', title: 'Done', status: 'done', ticketIds: [] },
  { id: 'col-blocked', title: 'Blocked', status: 'blocked', ticketIds: [] },
];

interface KanbanState {
  tickets: Record<string, KanbanTicket>;
  columns: KanbanColumn[];

  addTicket: (ticket: Omit<KanbanTicket, 'id' | 'createdAt' | 'updatedAt'>) => string;
  updateTicket: (id: string, updates: Partial<KanbanTicket>) => void;
  moveTicket: (ticketId: string, newStatus: TicketStatus) => void;
  deleteTicket: (ticketId: string) => void;
  addComment: (ticketId: string, comment: Omit<TicketComment, 'id' | 'timestamp'>) => void;
  addSubtask: (ticketId: string, subtask: Omit<Subtask, 'id'>) => void;
  toggleSubtask: (ticketId: string, subtaskId: string) => void;
}

export const useKanbanStore = create<KanbanState>((set) => ({
  tickets: {},
  columns: DEFAULT_COLUMNS.map((c) => ({ ...c, ticketIds: [...c.ticketIds] })),

  addTicket: (ticket) => {
    const id = nanoid();
    const now = Date.now();
    const newTicket: KanbanTicket = { ...ticket, id, createdAt: now, updatedAt: now };

    set((state) => ({
      tickets: { ...state.tickets, [id]: newTicket },
      columns: state.columns.map((col) =>
        col.status === newTicket.status
          ? { ...col, ticketIds: [...col.ticketIds, id] }
          : col,
      ),
    }));
    return id;
  },

  updateTicket: (id, updates) => {
    set((state) => {
      const existing = state.tickets[id];
      if (!existing) return state;
      return {
        tickets: {
          ...state.tickets,
          [id]: { ...existing, ...updates, updatedAt: Date.now() },
        },
      };
    });
  },

  moveTicket: (ticketId, newStatus) => {
    set((state) => {
      const ticket = state.tickets[ticketId];
      if (!ticket) return state;

      const oldStatus = ticket.status;
      if (oldStatus === newStatus) return state;

      return {
        tickets: {
          ...state.tickets,
          [ticketId]: { ...ticket, status: newStatus, updatedAt: Date.now() },
        },
        columns: state.columns.map((col) => {
          if (col.status === oldStatus) {
            return { ...col, ticketIds: col.ticketIds.filter((id) => id !== ticketId) };
          }
          if (col.status === newStatus) {
            return { ...col, ticketIds: [...col.ticketIds, ticketId] };
          }
          return col;
        }),
      };
    });
  },

  deleteTicket: (ticketId) => {
    set((state) => {
      const { [ticketId]: _, ...rest } = state.tickets;
      return {
        tickets: rest,
        columns: state.columns.map((col) => ({
          ...col,
          ticketIds: col.ticketIds.filter((id) => id !== ticketId),
        })),
      };
    });
  },

  addComment: (ticketId, comment) => {
    set((state) => {
      const ticket = state.tickets[ticketId];
      if (!ticket) return state;
      const newComment: TicketComment = {
        ...comment,
        id: nanoid(),
        timestamp: Date.now(),
      };
      return {
        tickets: {
          ...state.tickets,
          [ticketId]: {
            ...ticket,
            comments: [...ticket.comments, newComment],
            updatedAt: Date.now(),
          },
        },
      };
    });
  },

  addSubtask: (ticketId, subtask) => {
    set((state) => {
      const ticket = state.tickets[ticketId];
      if (!ticket) return state;
      const newSubtask: Subtask = { ...subtask, id: nanoid() };
      return {
        tickets: {
          ...state.tickets,
          [ticketId]: {
            ...ticket,
            subtasks: [...ticket.subtasks, newSubtask],
            updatedAt: Date.now(),
          },
        },
      };
    });
  },

  toggleSubtask: (ticketId, subtaskId) => {
    set((state) => {
      const ticket = state.tickets[ticketId];
      if (!ticket) return state;
      return {
        tickets: {
          ...state.tickets,
          [ticketId]: {
            ...ticket,
            subtasks: ticket.subtasks.map((st) =>
              st.id === subtaskId ? { ...st, done: !st.done } : st,
            ),
            updatedAt: Date.now(),
          },
        },
      };
    });
  },
}));
