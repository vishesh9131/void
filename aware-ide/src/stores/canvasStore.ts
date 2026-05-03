import { create } from 'zustand';
import {
  applyNodeChanges,
  applyEdgeChanges,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  type Connection,
} from '@xyflow/react';
import { nanoid } from 'nanoid';
import type { CanvasNode, CanvasEdge, CanvasNodeData } from '@/types/canvas';

interface CanvasState {
  nodes: CanvasNode[];
  edges: CanvasEdge[];

  addNode: (data: CanvasNodeData, position?: { x: number; y: number }) => string;
  removeNode: (id: string) => void;
  updateNode: (id: string, data: Partial<CanvasNodeData>) => void;
  addEdge: (edge: CanvasEdge) => void;
  removeEdge: (id: string) => void;
  onNodesChange: OnNodesChange<CanvasNode>;
  onEdgesChange: OnEdgesChange<CanvasEdge>;
  onConnect: OnConnect;
}

export const useCanvasStore = create<CanvasState>((set, get) => ({
  nodes: [],
  edges: [],

  addNode: (data, position = { x: 0, y: 0 }) => {
    const id = nanoid();
    const node: CanvasNode = { id, type: 'default', position, data };
    set((state) => ({ nodes: [...state.nodes, node] }));
    return id;
  },

  removeNode: (id) => {
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== id),
      edges: state.edges.filter((e) => e.source !== id && e.target !== id),
    }));
  },

  updateNode: (id, data) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...data } } : n,
      ),
    }));
  },

  addEdge: (edge) => {
    set((state) => ({ edges: [...state.edges, edge] }));
  },

  removeEdge: (id) => {
    set((state) => ({ edges: state.edges.filter((e) => e.id !== id) }));
  },

  onNodesChange: (changes) => {
    set((state) => ({ nodes: applyNodeChanges(changes, state.nodes) }));
  },

  onEdgesChange: (changes) => {
    set((state) => ({ edges: applyEdgeChanges(changes, state.edges) }));
  },

  onConnect: (connection: Connection) => {
    const edge: CanvasEdge = {
      id: nanoid(),
      source: connection.source,
      target: connection.target,
      sourceHandle: connection.sourceHandle,
      targetHandle: connection.targetHandle,
      data: { label: '', animated: false },
    };
    set((state) => ({ edges: [...state.edges, edge] }));
  },
}));
