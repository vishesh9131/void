import { create } from 'zustand';
import type { LLMConfig, LLMProvider } from '@/types/llm';

const DEFAULT_CONFIG: LLMConfig = {
  provider: 'vllm',
  model: '',
  baseUrl: 'https://vllm.corerec.online/v1',
  temperature: 0.7,
  maxTokens: 4096,
};

interface SettingsState {
  llmConfig: LLMConfig;
  theme: 'dark';
  projectPath: string | null;
  availableModels: string[];

  setProvider: (provider: LLMProvider) => void;
  setModel: (model: string) => void;
  setApiKey: (apiKey: string) => void;
  setBaseUrl: (baseUrl: string) => void;
  setTemperature: (temperature: number) => void;
  setMaxTokens: (maxTokens: number) => void;
  setProjectPath: (path: string | null) => void;
  setAvailableModels: (models: string[]) => void;
  updateConfig: (partial: Partial<LLMConfig>) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  llmConfig: { ...DEFAULT_CONFIG },
  theme: 'dark',
  projectPath: null,
  availableModels: [],

  setProvider: (provider) => {
    set((state) => ({ llmConfig: { ...state.llmConfig, provider } }));
  },

  setModel: (model) => {
    set((state) => ({ llmConfig: { ...state.llmConfig, model } }));
  },

  setApiKey: (apiKey) => {
    set((state) => ({ llmConfig: { ...state.llmConfig, apiKey } }));
  },

  setBaseUrl: (baseUrl) => {
    set((state) => ({ llmConfig: { ...state.llmConfig, baseUrl } }));
  },

  setTemperature: (temperature) => {
    set((state) => ({ llmConfig: { ...state.llmConfig, temperature } }));
  },

  setMaxTokens: (maxTokens) => {
    set((state) => ({ llmConfig: { ...state.llmConfig, maxTokens } }));
  },

  setProjectPath: (path) => {
    set({ projectPath: path });
  },

  setAvailableModels: (models) => {
    set({ availableModels: models });
  },

  updateConfig: (partial) => {
    set((state) => ({ llmConfig: { ...state.llmConfig, ...partial } }));
  },
}));
