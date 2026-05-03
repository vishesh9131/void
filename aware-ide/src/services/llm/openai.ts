import OpenAI from 'openai';
import type { ChatCompletionMessageParam, ChatCompletionTool } from 'openai/resources/chat/completions';
import { nanoid } from 'nanoid';
import type { ChatMessage, ToolCall } from '@/types/agents';
import type { CompletionRequest, LLMConfig, StreamChunk, ToolDefinition } from '@/types/llm';
import type { LLMService } from './provider';

function toOpenAIMessages(messages: ChatMessage[]): ChatCompletionMessageParam[] {
  return messages.map((msg): ChatCompletionMessageParam => {
    if (msg.role === 'tool') {
      return {
        role: 'tool',
        tool_call_id: msg.toolCalls?.[0]?.id ?? '',
        content: msg.content,
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

    if (msg.role === 'system') {
      return { role: 'system', content: msg.content };
    }

    if (msg.role === 'assistant') {
      return { role: 'assistant', content: msg.content };
    }

    return { role: 'user', content: msg.content };
  });
}

function toOpenAITools(tools: ToolDefinition[]): ChatCompletionTool[] {
  return tools.map((t) => ({
    type: 'function' as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters,
    },
  }));
}

export class OpenAIService implements LLMService {
  private client: OpenAI;
  private config: LLMConfig;

  constructor(config: LLMConfig) {
    this.config = config;
    this.client = new OpenAI({
      apiKey: config.apiKey,
      ...(config.baseUrl && { baseURL: config.baseUrl }),
      dangerouslyAllowBrowser: true,
    });
  }

  async complete(request: CompletionRequest): Promise<ChatMessage> {
    const messages = toOpenAIMessages(request.messages);

    try {
      const response = await this.client.chat.completions.create({
        model: request.config.model,
        messages,
        temperature: request.config.temperature,
        max_tokens: request.config.maxTokens,
        ...(request.tools?.length && { tools: toOpenAITools(request.tools) }),
      });

      const choice = response.choices[0];
      const toolCalls: ToolCall[] = (choice.message.tool_calls ?? []).map((tc) => ({
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
    } catch (err) {
      throw new Error(`OpenAI completion failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async *stream(request: CompletionRequest): AsyncGenerator<StreamChunk> {
    const messages = toOpenAIMessages(request.messages);

    try {
      const stream = await this.client.chat.completions.create({
        model: request.config.model,
        messages,
        temperature: request.config.temperature,
        max_tokens: request.config.maxTokens,
        stream: true,
        ...(request.tools?.length && { tools: toOpenAITools(request.tools) }),
      });

      // Accumulate tool call fragments across deltas
      const pendingTools = new Map<number, { id: string; name: string; args: string }>();

      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta;
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

        if (chunk.choices[0]?.finish_reason === 'tool_calls' || chunk.choices[0]?.finish_reason === 'stop') {
          // Flush accumulated tool calls
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
    } catch (err) {
      yield {
        type: 'error',
        content: `OpenAI stream error: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  }

  async listModels(): Promise<string[]> {
    try {
      const list = await this.client.models.list();
      return list.data.map((m) => m.id).sort();
    } catch (err) {
      throw new Error(`OpenAI listModels failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
}
