import { create } from 'zustand';
import { nanoid } from 'nanoid';
import type { EditorTab, FileEntry } from '@/types/editor';

interface EditorState {
  tabs: EditorTab[];
  activeTabId: string | null;
  fileTree: FileEntry | null;
  sidebarOpen: boolean;

  openFile: (path: string, content: string, language: string) => void;
  closeTab: (id: string) => void;
  setActiveTab: (id: string) => void;
  updateContent: (tabId: string, content: string) => void;
  markClean: (tabId: string) => void;
  setFileTree: (tree: FileEntry | null) => void;
  toggleSidebar: () => void;
  closeAllTabs: () => void;
}

export const useEditorStore = create<EditorState>((set, get) => ({
  tabs: [],
  activeTabId: null,
  fileTree: null,
  sidebarOpen: true,

  openFile: (path, content, language) => {
    const existing = get().tabs.find((t) => t.filePath === path);
    if (existing) {
      set({ activeTabId: existing.id });
      return;
    }

    const id = nanoid();
    const fileName = path.split('/').pop() ?? path;
    const tab: EditorTab = {
      id,
      filePath: path,
      fileName,
      content,
      language,
      isDirty: false,
      isActive: true,
    };

    set((state) => ({
      tabs: [
        ...state.tabs.map((t) => ({ ...t, isActive: false })),
        tab,
      ],
      activeTabId: id,
    }));
  },

  closeTab: (id) => {
    set((state) => {
      const idx = state.tabs.findIndex((t) => t.id === id);
      const remaining = state.tabs.filter((t) => t.id !== id);
      let nextActiveId: string | null = null;

      if (remaining.length > 0 && state.activeTabId === id) {
        const nextIdx = Math.min(idx, remaining.length - 1);
        nextActiveId = remaining[nextIdx].id;
      } else if (state.activeTabId !== id) {
        nextActiveId = state.activeTabId;
      }

      return {
        tabs: remaining.map((t) => ({ ...t, isActive: t.id === nextActiveId })),
        activeTabId: nextActiveId,
      };
    });
  },

  setActiveTab: (id) => {
    set((state) => ({
      tabs: state.tabs.map((t) => ({ ...t, isActive: t.id === id })),
      activeTabId: id,
    }));
  },

  updateContent: (tabId, content) => {
    set((state) => ({
      tabs: state.tabs.map((t) =>
        t.id === tabId ? { ...t, content, isDirty: true } : t,
      ),
    }));
  },

  markClean: (tabId) => {
    set((state) => ({
      tabs: state.tabs.map((t) =>
        t.id === tabId ? { ...t, isDirty: false } : t,
      ),
    }));
  },

  setFileTree: (tree) => {
    set({ fileTree: tree });
  },

  toggleSidebar: () => {
    set((state) => ({ sidebarOpen: !state.sidebarOpen }));
  },

  closeAllTabs: () => {
    set({ tabs: [], activeTabId: null });
  },
}));
