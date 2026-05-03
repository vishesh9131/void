export interface FileEntry {
  name: string;
  path: string;
  isDirectory: boolean;
  children?: FileEntry[];
}

export interface EditorTab {
  id: string;
  filePath: string;
  fileName: string;
  content: string;
  language: string;
  isDirty: boolean;
  isActive: boolean;
}

export type SplitDirection = 'horizontal' | 'vertical';
