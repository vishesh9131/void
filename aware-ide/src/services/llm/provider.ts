import type { ChatMessage } from '@/types/agents';
import type { CompletionRequest, LLMConfig, StreamChunk } from '@/types/llm';
import { AnthropicService } from './anthropic';
import { OpenAIService } from './openai';
import { VLLMService } from './vllm';

export interface LLMService {
  complete(request: CompletionRequest): Promise<ChatMessage>;
  stream(request: CompletionRequest): AsyncGenerator<StreamChunk>;
  listModels(): Promise<string[]>;
}

export function createLLMService(config: LLMConfig): LLMService {
  switch (config.provider) {
    case 'anthropic':
      return new AnthropicService(config);
    case 'openai':
      return new OpenAIService(config);
    case 'vllm':
      return new VLLMService(config);
  }
}
