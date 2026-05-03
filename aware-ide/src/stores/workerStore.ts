import { create } from 'zustand';
import { nanoid } from 'nanoid';
import type { WorkerAgent, AgentConfig, AgentStatus, ChatMessage } from '@/types/agents';

interface WorkerState {
  workers: Record<string, WorkerAgent>;

  addWorker: (name: string, config: AgentConfig) => string;
  removeWorker: (id: string) => void;
  updateWorker: (id: string, partial: Partial<WorkerAgent>) => void;
  setWorkerStatus: (id: string, status: AgentStatus) => void;
  addWorkerMessage: (workerId: string, message: ChatMessage) => void;
  setWorkerProgress: (id: string, progress: number) => void;
  assignTask: (workerId: string, task: string) => void;
}

export const useWorkerStore = create<WorkerState>((set) => ({
  workers: {},

  addWorker: (name, config) => {
    const id = nanoid();
    const worker: WorkerAgent = {
      id,
      name,
      status: 'idle',
      config,
      messages: [],
      assignedNodeIds: [],
      progress: 0,
    };
    set((state) => ({ workers: { ...state.workers, [id]: worker } }));
    return id;
  },

  removeWorker: (id) => {
    set((state) => {
      const { [id]: _, ...rest } = state.workers;
      return { workers: rest };
    });
  },

  updateWorker: (id, partial) => {
    set((state) => {
      const existing = state.workers[id];
      if (!existing) return state;
      return { workers: { ...state.workers, [id]: { ...existing, ...partial } } };
    });
  },

  setWorkerStatus: (id, status) => {
    set((state) => {
      const existing = state.workers[id];
      if (!existing) return state;
      return { workers: { ...state.workers, [id]: { ...existing, status } } };
    });
  },

  addWorkerMessage: (workerId, message) => {
    set((state) => {
      const existing = state.workers[workerId];
      if (!existing) return state;
      return {
        workers: {
          ...state.workers,
          [workerId]: { ...existing, messages: [...existing.messages, message] },
        },
      };
    });
  },

  setWorkerProgress: (id, progress) => {
    set((state) => {
      const existing = state.workers[id];
      if (!existing) return state;
      return { workers: { ...state.workers, [id]: { ...existing, progress } } };
    });
  },

  assignTask: (workerId, task) => {
    set((state) => {
      const existing = state.workers[workerId];
      if (!existing) return state;
      return {
        workers: {
          ...state.workers,
          [workerId]: { ...existing, currentTask: task, status: 'idle' },
        },
      };
    });
  },
}));
