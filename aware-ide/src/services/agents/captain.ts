import type { CaptainSession, WorkerAgent, ToolCall, ChatMessage, CaptainMode } from '@/types/agents';
import type { LLMConfig, ToolDefinition, StreamChunk } from '@/types/llm';
import type { KanbanTicket, TicketType, TicketPriority } from '@/types/kanban';
import type { ToolContext } from './tools';
import { ALL_TOOLS, executeTool } from './tools';
import { createLLMService } from '@/services/llm/provider';
import type { WorkerEvent } from './workerExec';

export interface CaptainContext extends ToolContext {
  workers: WorkerAgent[];
  kanbanTickets: KanbanTicket[];
}

// Captain-only tool definitions for managing workers and tasks
const planTasksDef: ToolDefinition = {
  name: 'plan_tasks',
  description: 'Break work into tasks and create kanban tickets for each.',
  parameters: {
    type: 'object',
    properties: {
      tasks: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            title: { type: 'string' },
            description: { type: 'string' },
            assignee: { type: 'string', description: 'Worker name or ID to assign' },
          },
          required: ['title', 'description'],
        },
      },
    },
    required: ['tasks'],
  },
};

const assignWorkerDef: ToolDefinition = {
  name: 'assign_worker',
  description: 'Assign a task to an existing worker agent.',
  parameters: {
    type: 'object',
    properties: {
      workerId: { type: 'string', description: 'Worker ID' },
      taskDescription: { type: 'string', description: 'Task to assign' },
    },
    required: ['workerId', 'taskDescription'],
  },
};

const reviewProgressDef: ToolDefinition = {
  name: 'review_progress',
  description: 'Review the status and progress of all workers.',
  parameters: {
    type: 'object',
    properties: {},
  },
};

const approveWorkDef: ToolDefinition = {
  name: 'approve_work',
  description: 'Mark a worker\'s current task as done.',
  parameters: {
    type: 'object',
    properties: {
      workerId: { type: 'string', description: 'Worker ID to approve' },
    },
    required: ['workerId'],
  },
};

const CAPTAIN_TOOLS: ToolDefinition[] = [planTasksDef, assignWorkerDef, reviewProgressDef, approveWorkDef];

function getCaptainTools(mode: CaptainMode): ToolDefinition[] {
  if (mode === 'ask') return [];
  return [...ALL_TOOLS, ...CAPTAIN_TOOLS];
}

const CAPTAIN_SYSTEM_PROMPT = `You are the Captain, a project manager agent in the Aware IDE.

Your responsibilities:
- Break down user requests into concrete tasks
- Create worker agents and assign tasks to them
- Create kanban tickets to track progress
- Monitor worker progress and review their output
- Coordinate between workers when tasks have dependencies
- Provide status updates to the user
- Approve completed work

When in "build" mode, use your tools to plan, delegate, and manage work.
When in "ask" mode, answer questions directly without using tools.

Be concise and action-oriented. Focus on getting work done efficiently.`;

const MAX_CAPTAIN_ROUNDS = 30;

export async function* runCaptainTurn(
  session: CaptainSession,
  config: LLMConfig,
  context: CaptainContext,
): AsyncGenerator<WorkerEvent> {
  const llm = createLLMService(config);
  const tools = getCaptainTools(session.mode);

  // Inject system prompt if not present
  if (session.messages.length === 0 || session.messages[0].role !== 'system') {
    session.messages.unshift({
      id: generateId(),
      role: 'system',
      content: CAPTAIN_SYSTEM_PROMPT,
      timestamp: Date.now(),
    });
  }

  // In ask mode, single completion with no tools
  if (session.mode === 'ask') {
    try {
      const stream = llm.stream({
        messages: session.messages,
        config,
        stream: true,
      });

      let textAccum = '';
      for await (const chunk of stream) {
        if (chunk.type === 'text') {
          textAccum += chunk.content;
          yield { type: 'text', content: chunk.content };
        } else if (chunk.type === 'thinking') {
          yield { type: 'thinking', content: chunk.content };
        } else if (chunk.type === 'error') {
          yield { type: 'error', message: chunk.content };
          return;
        }
      }

      if (textAccum) {
        session.messages.push({
          id: generateId(),
          role: 'assistant',
          content: textAccum,
          timestamp: Date.now(),
        });
      }

      yield { type: 'done' };
      return;
    } catch (err) {
      yield { type: 'error', message: err instanceof Error ? err.message : String(err) };
      return;
    }
  }

  // Build mode -- tool loop similar to worker but with captain-specific tools
  let rounds = 0;

  while (rounds < MAX_CAPTAIN_ROUNDS) {
    rounds++;
    const pendingToolCalls: ToolCall[] = [];
    let textAccum = '';

    try {
      const stream = llm.stream({
        messages: session.messages,
        config,
        tools,
        stream: true,
      });

      for await (const chunk of stream) {
        switch (chunk.type) {
          case 'text':
            textAccum += chunk.content;
            yield { type: 'text', content: chunk.content };
            break;
          case 'thinking':
            yield { type: 'thinking', content: chunk.content };
            break;
          case 'tool_call':
            if (chunk.toolCall) {
              pendingToolCalls.push(chunk.toolCall);
              yield { type: 'tool_call', call: chunk.toolCall };
            }
            break;
          case 'error':
            yield { type: 'error', message: chunk.content };
            return;
          case 'done':
            break;
        }
      }
    } catch (err) {
      yield { type: 'error', message: err instanceof Error ? err.message : String(err) };
      return;
    }

    if (textAccum || pendingToolCalls.length > 0) {
      session.messages.push({
        id: generateId(),
        role: 'assistant',
        content: textAccum,
        timestamp: Date.now(),
        toolCalls: pendingToolCalls.length > 0 ? pendingToolCalls : undefined,
      });
    }

    if (pendingToolCalls.length === 0) {
      break;
    }

    for (const tc of pendingToolCalls) {
      tc.status = 'running';
      let result: string;
      try {
        result = await executeCaptainTool(tc.name, tc.arguments, context);
        tc.status = 'done';
      } catch (err) {
        result = `Error: ${err instanceof Error ? err.message : String(err)}`;
        tc.status = 'error';
      }
      tc.result = result;

      yield { type: 'tool_result', callId: tc.id, result };

      session.messages.push({
        id: generateId(),
        role: 'tool',
        content: result,
        timestamp: Date.now(),
      });
    }
  }

  yield { type: 'done' };
}

async function executeCaptainTool(
  name: string,
  args: Record<string, unknown>,
  context: CaptainContext,
): Promise<string> {
  switch (name) {
    case 'plan_tasks': {
      const tasks = args.tasks as Array<{ title: string; description: string; assignee?: string }>;
      for (const task of tasks) {
        context.onKanbanAction({
          type: 'create_ticket',
          ticket: {
            title: task.title,
            description: task.description,
            type: 'task' as TicketType,
            priority: 'medium' as TicketPriority,
            status: 'todo',
            assigneeId: task.assignee,
          },
        });
      }
      return `Created ${tasks.length} task(s) on the kanban board.`;
    }

    case 'assign_worker': {
      const workerId = args.workerId as string;
      const taskDescription = args.taskDescription as string;
      const worker = context.workers.find((w) => w.id === workerId);
      if (!worker) return `Worker ${workerId} not found.`;
      worker.currentTask = taskDescription;
      worker.status = 'idle';
      return `Assigned task to ${worker.name}: ${taskDescription}`;
    }

    case 'review_progress': {
      if (context.workers.length === 0) return 'No workers active.';
      const lines = context.workers.map(
        (w) => `- ${w.name} (${w.id}): status=${w.status}, progress=${w.progress}%, task="${w.currentTask ?? 'none'}"`,
      );
      return `Worker status:\n${lines.join('\n')}`;
    }

    case 'approve_work': {
      const workerId = args.workerId as string;
      const worker = context.workers.find((w) => w.id === workerId);
      if (!worker) return `Worker ${workerId} not found.`;
      worker.status = 'done';
      worker.progress = 100;
      return `Approved work for ${worker.name}. Task marked done.`;
    }

    default:
      // Delegate to shared tool executor
      return executeTool(name, args, context);
  }
}

let idCounter = 0;
function generateId(): string {
  return `cap_${Date.now()}_${idCounter++}`;
}
