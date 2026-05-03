import { useCallback, type MouseEvent } from 'react';
import { X, FileCode, FileJson, FileText, File } from 'lucide-react';
import { useEditorStore } from '@/stores/editorStore';

function getFileIcon(fileName: string) {
  const ext = fileName.split('.').pop()?.toLowerCase();
  switch (ext) {
    case 'ts':
    case 'tsx':
    case 'js':
    case 'jsx':
      return <FileCode size={13} className="shrink-0 text-blue-400" />;
    case 'json':
      return <FileJson size={13} className="shrink-0 text-yellow-400" />;
    case 'md':
    case 'txt':
      return <FileText size={13} className="shrink-0 text-green-400" />;
    default:
      return <File size={13} className="shrink-0 text-aware-muted" />;
  }
}

export default function EditorTabs() {
  const tabs = useEditorStore((s) => s.tabs);
  const activeTabId = useEditorStore((s) => s.activeTabId);
  const setActiveTab = useEditorStore((s) => s.setActiveTab);
  const closeTab = useEditorStore((s) => s.closeTab);

  const handleClose = useCallback(
    (e: MouseEvent, id: string) => {
      e.stopPropagation();
      closeTab(id);
    },
    [closeTab],
  );

  const handleAuxClick = useCallback(
    (e: MouseEvent, id: string) => {
      if (e.button === 1) {
        e.preventDefault();
        closeTab(id);
      }
    },
    [closeTab],
  );

  if (tabs.length === 0) return null;

  return (
    <div className="flex h-9 shrink-0 items-center overflow-x-auto border-b border-aware-border bg-aware-panel scrollbar-none">
      {tabs.map((tab) => (
        <div
          key={tab.id}
          className={`group relative flex h-full cursor-pointer items-center gap-1.5 border-r border-aware-border px-3 text-xs transition-colors ${
            tab.id === activeTabId
              ? 'bg-aware-bg text-aware-text'
              : 'text-aware-muted hover:bg-aware-hover hover:text-aware-text'
          }`}
          onClick={() => setActiveTab(tab.id)}
          onAuxClick={(e) => handleAuxClick(e, tab.id)}
        >
          {getFileIcon(tab.fileName)}
          <span className="max-w-[120px] truncate">{tab.fileName}</span>
          {tab.isDirty && (
            <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-aware-accent" />
          )}
          <button
            onClick={(e) => handleClose(e, tab.id)}
            className="ml-1 shrink-0 rounded p-0.5 text-aware-muted opacity-0 hover:bg-aware-hover hover:text-aware-text group-hover:opacity-100 transition-all"
          >
            <X size={12} />
          </button>
          {tab.id === activeTabId && (
            <div className="absolute bottom-0 left-0 right-0 h-px bg-aware-accent" />
          )}
        </div>
      ))}
    </div>
  );
}
