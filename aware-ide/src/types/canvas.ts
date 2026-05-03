import type { Node, Edge } from '@xyflow/react';

export type NodeType =
  | 'worker'
  | 'agent'
  | 'block'
  | 'community'
  | 'group'
  | 'vault'
  | 'gate'
  | 'lens'
  | 'blueprint'
  | 'port'
  | 'trigger'
  | 'sandbox'
  | 'checkpoint'
  | 'canal'
  | 'evaluator';

export type NodeStatus = 'idle' | 'running' | 'done' | 'error';

export interface CanvasNodeData extends Record<string, unknown> {
  label: string;
  type: NodeType;
  status: NodeStatus;
  assignedTo?: string;
  content?: string;
  metadata: Record<string, unknown>;
}

export type CanvasNode = Node<CanvasNodeData>;

export interface CanvasEdgeData extends Record<string, unknown> {
  label: string;
  animated: boolean;
}

export type CanvasEdge = Edge<CanvasEdgeData>;

export type RelationType =
  | 'dependency'
  | 'data_flow'
  | 'communication'
  | 'trigger';
