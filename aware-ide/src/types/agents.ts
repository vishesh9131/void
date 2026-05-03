export type AgentRole = 'captain' | 'worker';

export type AgentStatus =
  | 'idle'
  | 'thinking'
  | 'working'
  | 'blocked'
  | 'done'
  | 'error';

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  result?: string;
  status: 'pending' | 'running' | 'done' | 'error';
}

export type ChatRole = 'user' | 'assistant' | 'system' | 'tool';

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  timestamp: number;
  toolCalls?: ToolCall[];
  thinking?: string;
}

export interface AgentConfig {
  id: string;
  name: string;
  role: AgentRole;
  model: string;
  provider: string;
  systemPrompt: string;
}

export interface WorkerAgent {
  id: string;
  name: string;
  status: AgentStatus;
  config: AgentConfig;
  currentTask?: string;
  messages: ChatMessage[];
  assignedNodeIds: string[];
  progress: number;
}

export type CaptainMode = 'ask' | 'build';

export interface CaptainSession {
  id: string;
  name: string;
  mode: CaptainMode;
  messages: ChatMessage[];
  createdAt: number;
  workerIds: string[];
}
