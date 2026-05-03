import type { ToolDefinition, ToolCall } from '@/types/llm';
import type { CanvasNodeData, CanvasEdgeData } from '@/types/canvas';
import type { KanbanTicket, TicketType, TicketPriority, TicketStatus } from '@/types/kanban';

export interface ToolContext {
  projectPath: string;
  onCanvasAction: (action: CanvasAction) => void;
  onKanbanAction: (action: KanbanAction) => void;
}

export type CanvasAction =
  | { type: 'create_block'; label: string; content: string }
  | { type: 'create_relation'; sourceId: string; targetId: string; label: string }
  | { type: 'create_worker_node'; name: string; task: string };

export type KanbanAction =
  | { type: 'create_ticket'; ticket: Omit<KanbanTicket, 'id' | 'createdAt' | 'updatedAt' | 'comments' | 'subtasks' | 'linkedNodeIds' | 'tags' | 'reporterId'> }
  | { type: 'update_ticket'; ticketId: string; status?: TicketStatus; comment?: string };

// -- Tool definitions for worker agents --

const writeFileDef: ToolDefinition = {
  name: 'write_file',
  description: 'Write content to a file at the given path.',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'File path relative to project root' },
      content: { type: 'string', description: 'File content to write' },
    },
    required: ['path', 'content'],
  },
};

const readFileDef: ToolDefinition = {
  name: 'read_file',
  description: 'Read the content of a file.',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'File path relative to project root' },
    },
    required: ['path'],
  },
};

const listDirectoryDef: ToolDefinition = {
  name: 'list_directory',
  description: 'List files and directories at the given path.',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'Directory path relative to project root' },
    },
    required: ['path'],
  },
};

const editFileDef: ToolDefinition = {
  name: 'edit_file',
  description: 'Find and replace text in a file.',
  parameters: {
    type: 'object',
    properties: {
      path: { type: 'string', description: 'File path relative to project root' },
      search: { type: 'string', description: 'Text to find' },
      replace: { type: 'string', description: 'Replacement text' },
    },
    required: ['path', 'search', 'replace'],
  },
};

const runTerminalDef: ToolDefinition = {
  name: 'run_terminal',
  description: 'Run a shell command and return output.',
  parameters: {
    type: 'object',
    properties: {
      command: { type: 'string', description: 'Shell command to execute' },
    },
    required: ['command'],
  },
};

const grepWorkspaceDef: ToolDefinition = {
  name: 'grep_workspace',
  description: 'Search files for a pattern.',
  parameters: {
    type: 'object',
    properties: {
      pattern: { type: 'string', description: 'Search pattern (regex)' },
      path: { type: 'string', description: 'Optional subdirectory to search in' },
    },
    required: ['pattern'],
  },
};

const createBlockDef: ToolDefinition = {
  name: 'create_block',
  description: 'Create a code block node on the canvas.',
  parameters: {
    type: 'object',
    properties: {
      label: { type: 'string', description: 'Block label' },
      content: { type: 'string', description: 'Block content (code or text)' },
    },
    required: ['label', 'content'],
  },
};

const createWorkerDef: ToolDefinition = {
  name: 'create_worker',
  description: 'Spawn a sub-worker agent with a task.',
  parameters: {
    type: 'object',
    properties: {
      name: { type: 'string', description: 'Worker name' },
      task: { type: 'string', description: 'Task description for the worker' },
    },
    required: ['name', 'task'],
  },
};

const sendToWorkerDef: ToolDefinition = {
  name: 'send_to_worker',
  description: 'Send a message to an existing worker agent.',
  parameters: {
    type: 'object',
    properties: {
      workerId: { type: 'string', description: 'Target worker ID' },
      message: { type: 'string', description: 'Message content' },
    },
    required: ['workerId', 'message'],
  },
};

const createRelationDef: ToolDefinition = {
  name: 'create_relation',
  description: 'Create an edge between two nodes on the canvas.',
  parameters: {
    type: 'object',
    properties: {
      sourceId: { type: 'string', description: 'Source node ID' },
      targetId: { type: 'string', description: 'Target node ID' },
      label: { type: 'string', description: 'Edge label' },
    },
    required: ['sourceId', 'targetId', 'label'],
  },
};

const createTicketDef: ToolDefinition = {
  name: 'create_ticket',
  description: 'Create a kanban ticket.',
  parameters: {
    type: 'object',
    properties: {
      title: { type: 'string', description: 'Ticket title' },
      description: { type: 'string', description: 'Ticket description' },
      type: { type: 'string', enum: ['task', 'bug', 'feature', 'issue', 'improvement'] },
      priority: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
    },
    required: ['title', 'description', 'type', 'priority'],
  },
};

const updateTicketDef: ToolDefinition = {
  name: 'update_ticket',
  description: 'Update a kanban ticket status or add a comment.',
  parameters: {
    type: 'object',
    properties: {
      ticketId: { type: 'string', description: 'Ticket ID' },
      status: { type: 'string', enum: ['backlog', 'todo', 'in_progress', 'review', 'done', 'blocked'] },
      comment: { type: 'string', description: 'Comment to add' },
    },
    required: ['ticketId'],
  },
};

export const ALL_TOOLS: ToolDefinition[] = [
  writeFileDef,
  readFileDef,
  listDirectoryDef,
  editFileDef,
  runTerminalDef,
  grepWorkspaceDef,
  createBlockDef,
  createWorkerDef,
  sendToWorkerDef,
  createRelationDef,
  createTicketDef,
  updateTicketDef,
];

export async function executeTool(
  name: string,
  args: Record<string, unknown>,
  context: ToolContext,
): Promise<string> {
  try {
    switch (name) {
      case 'write_file':
        return stubFileOp('write', args.path as string, context);
      case 'read_file':
        return stubFileOp('read', args.path as string, context);
      case 'list_directory':
        return stubFileOp('list', args.path as string, context);
      case 'edit_file':
        return stubFileOp('edit', args.path as string, context);
      case 'run_terminal':
        return stubTerminal(args.command as string, context);
      case 'grep_workspace':
        return stubGrep(args.pattern as string, args.path as string | undefined, context);
      case 'create_block':
        context.onCanvasAction({
          type: 'create_block',
          label: args.label as string,
          content: args.content as string,
        });
        return `Block "${args.label}" created on canvas.`;
      case 'create_worker':
        context.onCanvasAction({
          type: 'create_worker_node',
          name: args.name as string,
          task: args.task as string,
        });
        return `Worker "${args.name}" spawned with task: ${args.task}`;
      case 'send_to_worker':
        return `Message sent to worker ${args.workerId}: ${args.message}`;
      case 'create_relation':
        context.onCanvasAction({
          type: 'create_relation',
          sourceId: args.sourceId as string,
          targetId: args.targetId as string,
          label: args.label as string,
        });
        return `Relation created: ${args.sourceId} -> ${args.targetId} [${args.label}]`;
      case 'create_ticket':
        context.onKanbanAction({
          type: 'create_ticket',
          ticket: {
            title: args.title as string,
            description: args.description as string,
            type: args.type as TicketType,
            priority: args.priority as TicketPriority,
            status: 'todo',
            assigneeId: undefined,
          },
        });
        return `Ticket created: "${args.title}"`;
      case 'update_ticket':
        context.onKanbanAction({
          type: 'update_ticket',
          ticketId: args.ticketId as string,
          status: args.status as TicketStatus | undefined,
          comment: args.comment as string | undefined,
        });
        return `Ticket ${args.ticketId} updated.`;
      default:
        return `Unknown tool: ${name}`;
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return `Tool error (${name}): ${msg}`;
  }
}

// Stub implementations -- actual fs/terminal access routed through Electron IPC or browser sandbox
function stubFileOp(op: string, path: string, context: ToolContext): string {
  const fullPath = `${context.projectPath}/${path}`;
  return `[stub] ${op} on ${fullPath} -- routed through host backend`;
}

function stubTerminal(command: string, _context: ToolContext): string {
  return `[stub] terminal: ${command} -- routed through host backend`;
}

function stubGrep(pattern: string, path: string | undefined, context: ToolContext): string {
  const target = path ? `${context.projectPath}/${path}` : context.projectPath;
  return `[stub] grep "${pattern}" in ${target} -- routed through host backend`;
}
