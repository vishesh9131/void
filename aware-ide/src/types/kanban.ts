export type TicketPriority = 'critical' | 'high' | 'medium' | 'low';

export type TicketStatus =
  | 'backlog'
  | 'todo'
  | 'in_progress'
  | 'review'
  | 'done'
  | 'blocked';

export type TicketType = 'task' | 'bug' | 'feature' | 'issue' | 'improvement';

export interface Subtask {
  id: string;
  title: string;
  done: boolean;
}

export interface TicketComment {
  id: string;
  author: string;
  content: string;
  timestamp: number;
}

export interface KanbanTicket {
  id: string;
  title: string;
  description: string;
  type: TicketType;
  priority: TicketPriority;
  status: TicketStatus;
  assigneeId?: string;
  reporterId: string;
  createdAt: number;
  updatedAt: number;
  tags: string[];
  subtasks: Subtask[];
  comments: TicketComment[];
  linkedNodeIds: string[];
}

export interface KanbanColumn {
  id: string;
  title: string;
  status: TicketStatus;
  ticketIds: string[];
}
