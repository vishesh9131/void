import { nanoid } from 'nanoid';
import type { ChatMessage, ToolCall } from '@/types/agents';
import type { CompletionRequest, LLMConfig, StreamChunk, ToolDefinition } from '@/types/llm';
import type { LLMService } from './provider';

const DEFAULT_BASE_URL = 'https://vllm.corerec.online/v1';

interface VLLMChatMessage {
  role: string;
  content: string | null;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
}

interface VLLMToolDef {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

function toVLLMMessages(messages: ChatMessage[]): VLLMChatMessage[] {
  return messages.map((msg): VLLMChatMessage => {
    if (msg.role === 'tool') {
      return {
        role: 'tool',
        content: msg.content,
        tool_call_id: msg.toolCalls?.[0]?.id ?? '',
      };
    }

    if (msg.role === 'assistant' && msg.toolCalls?.length) {
      return {
        role: 'assistant',
        content: msg.content || null,
        tool_calls: msg.toolCalls.map((tc) => ({
          id: tc.id,
          type: 'function' as const,
          function: {
            name: tc.name,
            arguments: JSON.stringify(tc.arguments),
          },
        })),
      };
    }

    return { role: msg.role, content: msg.content };
  });
}

function toVLLMTools(tools: ToolDefinition[]): VLLMToolDef[] {
  return tools.map((t) => ({
    type: 'function' as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    },
  }));
}

export class VLLMService implements LLMService {
  private baseUrl: string;
  private config: LLMConfig;

  constructor(config: LLMConfig) {
    this.config = config;
    this.baseUrl = config.baseUrl || DEFAULT_BASE_URL;
  }

  async complete(request: CompletionRequest): Promise<ChatMessage> {
    const body: Record<string, unknown> = {
      model: request.config.model,
      messages: toVLLMMessages(request.messages),
      temperature: request.config.temperature,
      max_tokens: request.config.maxTokens,
      stream: false,
    };
    if (request.tools?.length) {
      body.tools = toVLLMTools(request.tools);
    }

    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch (err) {
      throw new Error(`vLLM request failed: ${err instanceof Error ? err.message : String(err)}`);
    }

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`vLLM returned ${res.status}: ${text}`);
    }

    const data = await res.json();
    const choice = data.choices?.[0];
    if (!choice) {
      throw new Error('vLLM returned no choices');
    }

    const toolCalls: ToolCall[] = (choice.message.tool_calls ?? []).map((tc: { id: string; function: { name: string; arguments: string } }) => ({
      id: tc.id,
      name: tc.function.name,
      arguments: JSON.parse(tc.function.arguments || '{}'),
      status: 'pending' as const,
    }));

    return {
      id: nanoid(),
      role: 'assistant',
      content: choice.message.content ?? '',
      timestamp: Date.now(),
      ...(toolCalls.length && { toolCalls }),
    };
  }

  async *stream(request: CompletionRequest): AsyncGenerator<StreamChunk> {
    const body: Record<string, unknown> = {
      model: request.config.model,
      messages: toVLLMMessages(request.messages),
      temperature: request.config.temperature,
      max_tokens: request.config.maxTokens,
      stream: true,
    };
    if (request.tools?.length) {
      body.tools = toVLLMTools(request.tools);
    }

    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    } catch (err) {
      yield {
        type: 'error',
        content: `vLLM stream request failed: ${err instanceof Error ? err.message : String(err)}`,
      };
      return;
    }

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      yield { type: 'error', content: `vLLM returned ${res.status}: ${text}` };
      return;
    }

    if (!res.body) {
      yield { type: 'error', content: 'vLLM response has no body' };
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Accumulate tool call fragments across SSE events
    const pendingTools = new Map<number, { id: string; name: string; args: string }>();

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        // Keep the last incomplete line in the buffer
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith(':')) continue;

          if (!trimmed.startsWith('data: ')) continue;
          const payload = trimmed.slice(6);

          if (payload === '[DONE]') {
            // Flush any remaining tool calls
            for (const [, pending] of pendingTools) {
              let args: Record<string, unknown> = {};
              try {
                args = JSON.parse(pending.args || '{}');
              } catch {
                // partial JSON
              }
              yield {
                type: 'tool_call',
                content: '',
                toolCall: {
                  id: pending.id,
                  name: pending.name,
                  arguments: args,
                  status: 'pending',
                },
              };
            }
            pendingTools.clear();
            yield { type: 'done', content: '' };
            return;
          }

          let chunk: Record<string, unknown>;
          try {
            chunk = JSON.parse(payload);
          } catch {
            continue;
          }

          const choices = chunk.choices as Array<{
            delta?: {
              content?: string;
              tool_calls?: Array<{
                index: number;
                id?: string;
                function?: { name?: string; arguments?: string };
              }>;
            };
            finish_reason?: string;
          }> | undefined;

          const delta = choices?.[0]?.delta;
          if (!delta) continue;

          if (delta.content) {
            yield { type: 'text', content: delta.content };
          }

          if (delta.tool_calls) {
            for (const tc of delta.tool_calls) {
              const idx = tc.index;
              if (!pendingTools.has(idx)) {
                pendingTools.set(idx, { id: tc.id ?? '', name: tc.function?.name ?? '', args: '' });
              }
              const pending = pendingTools.get(idx)!;
              if (tc.id) pending.id = tc.id;
              if (tc.function?.name) pending.name = tc.function.name;
              if (tc.function?.arguments) pending.args += tc.function.arguments;
            }
          }

          const finishReason = choices?.[0]?.finish_reason;
          if (finishReason === 'tool_calls' || finishReason === 'stop') {
            for (const [, pending] of pendingTools) {
              let args: Record<string, unknown> = {};
              try {
                args = JSON.parse(pending.args || '{}');
              } catch {
                // partial JSON
              }
              yield {
                type: 'tool_call',
                content: '',
                toolCall: {
                  id: pending.id,
                  name: pending.name,
                  arguments: args,
                  status: 'pending',
                },
              };
            }
            pendingTools.clear();
            yield { type: 'done', content: '' };
          }
        }
      }
    } catch (err) {
      yield {
        type: 'error',
        content: `vLLM stream parse error: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  }

  async listModels(): Promise<string[]> {
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}/models`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });
    } catch (err) {
      throw new Error(`vLLM listModels failed: ${err instanceof Error ? err.message : String(err)}`);
    }

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`vLLM listModels returned ${res.status}: ${text}`);
    }

    const data = await res.json();
    return (data.data ?? []).map((m: { id: string }) => m.id).sort();
  }
}
