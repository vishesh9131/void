import type { WorkerAgent, ToolCall, ChatMessage } from '@/types/agents';
import type { LLMConfig, StreamChunk } from '@/types/llm';
import type { ToolContext } from './tools';
import { ALL_TOOLS, executeTool } from './tools';
import { createLLMService } from '@/services/llm/provider';

export type WorkerEvent =
  | { type: 'text'; content: string }
  | { type: 'tool_call'; call: ToolCall }
  | { type: 'tool_result'; callId: string; result: string }
  | { type: 'thinking'; content: string }
  | { type: 'done' }
  | { type: 'error'; message: string };

const MAX_TOOL_ROUNDS = 25;

export async function* runWorkerTurn(
  worker: WorkerAgent,
  config: LLMConfig,
  context: ToolContext,
): AsyncGenerator<WorkerEvent> {
  const llm = createLLMService(config);
  let rounds = 0;

  while (rounds < MAX_TOOL_ROUNDS) {
    rounds++;
    const pendingToolCalls: ToolCall[] = [];
    let textAccum = '';
    let hadToolCalls = false;

    try {
      const stream = llm.stream({
        messages: worker.messages,
        config,
        tools: ALL_TOOLS,
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
      const msg = err instanceof Error ? err.message : String(err);
      yield { type: 'error', message: msg };
      return;
    }

    // Append assistant text to conversation
    if (textAccum) {
      const assistantMsg: ChatMessage = {
        id: generateId(),
        role: 'assistant',
        content: textAccum,
        timestamp: Date.now(),
        toolCalls: pendingToolCalls.length > 0 ? pendingToolCalls : undefined,
      };
      worker.messages.push(assistantMsg);
    }

    if (pendingToolCalls.length === 0) {
      break;
    }

    hadToolCalls = true;

    // Execute each tool call and feed results back
    for (const tc of pendingToolCalls) {
      tc.status = 'running';
      let result: string;
      try {
        result = await executeTool(tc.name, tc.arguments, context);
        tc.status = 'done';
      } catch (err) {
        result = `Error: ${err instanceof Error ? err.message : String(err)}`;
        tc.status = 'error';
      }
      tc.result = result;

      yield { type: 'tool_result', callId: tc.id, result };

      const toolMsg: ChatMessage = {
        id: generateId(),
        role: 'tool',
        content: result,
        timestamp: Date.now(),
      };
      worker.messages.push(toolMsg);
    }

    if (!hadToolCalls) {
      break;
    }
  }

  yield { type: 'done' };
}

let idCounter = 0;
function generateId(): string {
  return `msg_${Date.now()}_${idCounter++}`;
}
