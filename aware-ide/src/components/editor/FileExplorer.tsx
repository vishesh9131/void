import { useState, useCallback } from 'react';
import { ChevronRight, ChevronDown, Folder, FolderOpen, FileCode, FileJson, FileText, File } from 'lucide-react';
import { useEditorStore } from '@/stores/editorStore';
import type { FileEntry } from '@/types/editor';

function getFileIcon(name: string) {
  const ext = name.split('.').pop()?.toLowerCase();
  switch (ext) {
    case 'ts':
    case 'tsx':
    case 'js':
    case 'jsx':
      return <FileCode size={14} className="shrink-0 text-blue-400" />;
    case 'json':
      return <FileJson size={14} className="shrink-0 text-yellow-400" />;
    case 'md':
    case 'txt':
      return <FileText size={14} className="shrink-0 text-green-400" />;
    default:
      return <File size={14} className="shrink-0 text-aware-muted" />;
  }
}

interface TreeNodeProps {
  entry: FileEntry;
  depth: number;
  onFileClick: (entry: FileEntry) => void;
}

function TreeNode({ entry, depth, onFileClick }: TreeNodeProps) {
  const [expanded, setExpanded] = useState(depth < 1);

  const handleClick = useCallback(() => {
    if (entry.isDirectory) {
      setExpanded((prev) => !prev);
    } else {
      onFileClick(entry);
    }
  }, [entry, onFileClick]);

  return (
    <div>
      <div
        className="flex cursor-pointer items-center gap-1.5 px-2 py-1 text-xs hover:bg-aware-hover transition-colors"
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        onClick={handleClick}
      >
        {entry.isDirectory ? (
          <>
            {expanded ? (
              <ChevronDown size={12} className="shrink-0 text-aware-muted" />
            ) : (
              <ChevronRight size={12} className="shrink-0 text-aware-muted" />
            )}
            {expanded ? (
              <FolderOpen size={14} className="shrink-0 text-aware-accent" />
            ) : (
              <Folder size={14} className="shrink-0 text-aware-accent" />
            )}
          </>
        ) : (
          <>
            <span className="w-3 shrink-0" />
            {getFileIcon(entry.name)}
          </>
        )}
        <span className="truncate text-aware-text">{entry.name}</span>
      </div>

      {entry.isDirectory && expanded && entry.children && (
        <div>
          {entry.children.map((child) => (
            <TreeNode
              key={child.path}
              entry={child}
              depth={depth + 1}
              onFileClick={onFileClick}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function FileExplorer() {
  const fileTree = useEditorStore((s) => s.fileTree);
  const openFile = useEditorStore((s) => s.openFile);

  const handleFileClick = useCallback(
    (entry: FileEntry) => {
      const ext = entry.name.split('.').pop()?.toLowerCase() ?? '';
      const langMap: Record<string, string> = {
        ts: 'typescript',
        tsx: 'typescriptreact',
        js: 'javascript',
        jsx: 'javascriptreact',
        json: 'json',
        md: 'markdown',
        css: 'css',
        html: 'html',
        py: 'python',
        rs: 'rust',
        go: 'go',
      };
      const language = langMap[ext] || 'plaintext';
      openFile(entry.path, '', language);
    },
    [openFile],
  );

  if (!fileTree) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-4 text-center">
        <Folder size={32} className="text-aware-muted/50" />
        <p className="text-xs text-aware-muted">
          No project open
        </p>
        <button className="rounded-lg bg-aware-accent/10 px-3 py-1.5 text-xs font-medium text-aware-accent hover:bg-aware-accent/20 transition-colors">
          Open Project
        </button>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto py-1">
      <div className="mb-1 px-3 py-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-aware-muted">
          Explorer
        </span>
      </div>
      {fileTree.children?.map((child) => (
        <TreeNode
          key={child.path}
          entry={child}
          depth={0}
          onFileClick={handleFileClick}
        />
      ))}
    </div>
  );
}
