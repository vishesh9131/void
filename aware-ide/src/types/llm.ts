import type { ChatMessage, ToolCall } from './agents';

export type LLMProvider = 'anthropic' | 'openai' | 'vllm';

export interface LLMConfig {
  provider: LLMProvider;
  model: string;
  apiKey?: string;
  baseUrl?: string;
  temperature: number;
  maxTokens: number;
}

export type StreamChunkType = 'text' | 'tool_call' | 'thinking' | 'done' | 'error';

export interface StreamChunk {
  type: StreamChunkType;
  content: string;
  toolCall?: ToolCall;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface CompletionRequest {
  messages: ChatMessage[];
  config: LLMConfig;
  tools?: ToolDefinition[];
  stream: boolean;
}
