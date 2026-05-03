import Anthropic from '@anthropic-ai/sdk';
import type { MessageParam, Tool, ContentBlock, TextBlock, ToolUseBlock, ContentBlockDeltaEvent } from '@anthropic-ai/sdk/resources/messages';
import { nanoid } from 'nanoid';
import type { ChatMessage, ToolCall } from '@/types/agents';
import type { CompletionRequest, LLMConfig, StreamChunk, ToolDefinition } from '@/types/llm';
import type { LLMService } from './provider';

function toAnthropicMessages(messages: ChatMessage[]): { system: string | undefined; messages: MessageParam[] } {
  let system: string | undefined;
  const mapped: MessageParam[] = [];

  for (const msg of messages) {
    if (msg.role === 'system') {
      system = msg.content;
      continue;
    }

    if (msg.role === 'tool') {
      // Tool results get attached as user messages with tool_result content
      mapped.push({
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: msg.toolCalls?.[0]?.id ?? '',
          content: msg.content,
        }],
      });
      continue;
    }

    if (msg.role === 'assistant' && msg.toolCalls?.length) {
      const content: (TextBlock | ToolUseBlock)[] = [];
      if (msg.content) {
        content.push({ type: 'text', text: msg.content });
      }
      for (const tc of msg.toolCalls) {
        content.push({
          type: 'tool_use',
          id: tc.id,
          name: tc.name,
          input: tc.arguments,
        });
      }
      mapped.push({ role: 'assistant', content });
      continue;
    }

    mapped.push({
      role: msg.role === 'assistant' ? 'assistant' : 'user',
      content: msg.content,
    });
  }

  return { system, messages: mapped };
}

function toAnthropicTools(tools: ToolDefinition[]): Tool[] {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    input_schema: t.parameters as Tool['input_schema'],
  }));
}

function extractToolCalls(blocks: ContentBlock[]): ToolCall[] {
  return blocks
    .filter((b): b is ToolUseBlock => b.type === 'tool_use')
    .map((b) => ({
      id: b.id,
      name: b.name,
      arguments: b.input as Record<string, unknown>,
      status: 'pending' as const,
    }));
}

export class AnthropicService implements LLMService {
  private client: Anthropic;
  private config: LLMConfig;

  constructor(config: LLMConfig) {
    this.config = config;
    this.client = new Anthropic({
      apiKey: config.apiKey,
      ...(config.baseUrl && { baseURL: config.baseUrl }),
    });
  }

  async complete(request: CompletionRequest): Promise<ChatMessage> {
    const { system, messages } = toAnthropicMessages(request.messages);

    try {
      const response = await this.client.messages.create({
        model: request.config.model,
        max_tokens: request.config.maxTokens,
        temperature: request.config.temperature,
        ...(system && { system }),
        messages,
        ...(request.tools?.length && { tools: toAnthropicTools(request.tools) }),
      });

      const textParts = response.content
        .filter((b): b is TextBlock => b.type === 'text')
        .map((b) => b.text);

      const toolCalls = extractToolCalls(response.content);

      return {
        id: nanoid(),
        role: 'assistant',
        content: textParts.join(''),
        timestamp: Date.now(),
        ...(toolCalls.length && { toolCalls }),
      };
    } catch (err) {
      throw new Error(`Anthropic completion failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async *stream(request: CompletionRequest): AsyncGenerator<StreamChunk> {
    const { system, messages } = toAnthropicMessages(request.messages);

    try {
      const stream = this.client.messages.stream({
        model: request.config.model,
        max_tokens: request.config.maxTokens,
        temperature: request.config.temperature,
        ...(system && { system }),
        messages,
        ...(request.tools?.length && { tools: toAnthropicTools(request.tools) }),
      });

      // Track in-progress tool calls across deltas
      let currentToolId = '';
      let currentToolName = '';
      let toolInputJson = '';

      for await (const event of stream) {
        if (event.type === 'content_block_start') {
          const block = event.content_block;
          if (block.type === 'tool_use') {
            currentToolId = block.id;
            currentToolName = block.name;
            toolInputJson = '';
          }
        } else if (event.type === 'content_block_delta') {
          const delta = (event as ContentBlockDeltaEvent).delta;
          if (delta.type === 'text_delta') {
            yield { type: 'text', content: delta.text };
          } else if (delta.type === 'input_json_delta') {
            toolInputJson += delta.partial_json;
          }
        } else if (event.type === 'content_block_stop') {
          if (currentToolId) {
            let args: Record<string, unknown> = {};
            try {
              args = JSON.parse(toolInputJson || '{}');
            } catch {
              // partial/malformed JSON -- pass empty args
            }
            yield {
              type: 'tool_call',
              content: '',
              toolCall: {
                id: currentToolId,
                name: currentToolName,
                arguments: args,
                status: 'pending',
              },
            };
            currentToolId = '';
            currentToolName = '';
            toolInputJson = '';
          }
        } else if (event.type === 'message_stop') {
          yield { type: 'done', content: '' };
        }
      }
    } catch (err) {
      yield {
        type: 'error',
        content: `Anthropic stream error: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  }

  async listModels(): Promise<string[]> {
    // Anthropic doesn't expose a models list endpoint; return known models
    return [
      'claude-sonnet-4-20250514',
      'claude-3-5-sonnet-20241022',
      'claude-3-5-haiku-20241022',
      'claude-3-opus-20240229',
    ];
  }
}
