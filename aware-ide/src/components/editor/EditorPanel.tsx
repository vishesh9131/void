import { useCallback } from 'react';
import Editor from '@monaco-editor/react';
import { useEditorStore } from '@/stores/editorStore';
import EditorTabs from './EditorTabs';
import WelcomeScreen from './WelcomeScreen';

export default function EditorPanel() {
  const tabs = useEditorStore((s) => s.tabs);
  const activeTabId = useEditorStore((s) => s.activeTabId);
  const updateContent = useEditorStore((s) => s.updateContent);

  const activeTab = tabs.find((t) => t.id === activeTabId);

  const handleEditorChange = useCallback(
    (value: string | undefined) => {
      if (activeTabId && value !== undefined) {
        updateContent(activeTabId, value);
      }
    },
    [activeTabId, updateContent],
  );

  if (tabs.length === 0) {
    return <WelcomeScreen />;
  }

  return (
    <div className="flex h-full flex-col bg-aware-bg">
      <EditorTabs />
      <div className="flex-1 overflow-hidden">
        {activeTab ? (
          <Editor
            key={activeTab.id}
            theme="vs-dark"
            language={activeTab.language}
            value={activeTab.content}
            onChange={handleEditorChange}
            options={{
              fontSize: 13,
              fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              padding: { top: 12 },
              lineNumbers: 'on',
              renderLineHighlight: 'line',
              bracketPairColorization: { enabled: true },
              automaticLayout: true,
              tabSize: 2,
            }}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-aware-muted">
            Select a tab
          </div>
        )}
      </div>
    </div>
  );
}
