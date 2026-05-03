import { useState } from 'react';
import ActivityBar, { type ViewId } from './ActivityBar';
import StatusBar from './StatusBar';
import Mapper from '@/components/canvas/Mapper';
import CaptainPanel from '@/components/captain/CaptainPanel';
import EditorPanel from '@/components/editor/EditorPanel';
import FileExplorer from '@/components/editor/FileExplorer';
import KanbanBoard from '@/components/kanban/KanbanBoard';
import SettingsPanel from '@/components/settings/SettingsPanel';
import { useCaptainRunner } from '@/components/captain/CaptainRunner';

export default function AppLayout() {
  const [activeView, setActiveView] = useState<ViewId>('canvas');

  // Initialise the captain runner so agent loop is available app-wide
  useCaptainRunner();

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-aware-bg">
      <div className="flex flex-1 overflow-hidden">
        <ActivityBar activeView={activeView} onViewChange={setActiveView} />

        <main className="flex-1 overflow-hidden">
          {activeView === 'canvas' && <Mapper />}
          {activeView === 'captain' && <CaptainPanel />}
          {activeView === 'editor' && (
            <div className="flex h-full">
              <div className="w-60 shrink-0 border-r border-aware-border bg-aware-panel overflow-hidden">
                <FileExplorer />
              </div>
              <div className="flex-1 overflow-hidden">
                <EditorPanel />
              </div>
            </div>
          )}
          {activeView === 'kanban' && <KanbanBoard />}
          {activeView === 'settings' && <SettingsPanel />}
        </main>
      </div>

      <StatusBar onNavigate={setActiveView} />
    </div>
  );
}
