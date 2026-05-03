import { memo, useCallback } from 'react';
import { Bot, Code, Shield, SquareTerminal, Users } from 'lucide-react';
import { useCanvasStore } from '@/stores/canvasStore';
import type { CanvasNodeData, NodeType } from '@/types/canvas';

interface ToolbarItem {
  type: NodeType;
  label: string;
  icon: React.ReactNode;
}

const items: ToolbarItem[] = [
  { type: 'worker', label: 'Worker', icon: <Bot size={15} /> },
  { type: 'block', label: 'Block', icon: <Code size={15} /> },
  { type: 'gate', label: 'Gate', icon: <Shield size={15} /> },
  { type: 'sandbox', label: 'Sandbox', icon: <SquareTerminal size={15} /> },
  { type: 'group', label: 'Group', icon: <Users size={15} /> },
];

function NodeToolbar() {
  const addNode = useCanvasStore((s) => s.addNode);

  const handleAdd = useCallback(
    (type: NodeType, label: string) => {
      const x = 100 + Math.random() * 400;
      const y = 100 + Math.random() * 300;
      const data: CanvasNodeData = {
        label: `New ${label}`,
        type,
        status: 'idle',
        metadata: {},
      };
      addNode(data, { x, y });
    },
    [addNode],
  );

  return (
    <div className="flex items-center gap-1 bg-aware-panel border border-aware-border rounded-lg px-2 py-1.5 shadow-lg">
      {items.map((item) => (
        <button
          key={item.type}
          onClick={() => handleAdd(item.type, item.label)}
          title={`Add ${item.label}`}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-aware-muted hover:text-aware-text hover:bg-aware-hover transition-colors text-xs font-medium"
        >
          {item.icon}
          <span className="hidden sm:inline">{item.label}</span>
        </button>
      ))}
    </div>
  );
}

export default memo(NodeToolbar);
