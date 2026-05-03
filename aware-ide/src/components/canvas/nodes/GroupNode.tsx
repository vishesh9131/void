import { memo } from 'react';
import { type NodeProps } from '@xyflow/react';
import { Users } from 'lucide-react';
import type { CanvasNodeData } from '@/types/canvas';

function GroupNode({ data }: NodeProps) {
  const nodeData = data as unknown as CanvasNodeData;

  return (
    <div className="min-w-[300px] min-h-[200px] rounded-xl border-2 border-dashed border-aware-border bg-aware-surface/30 backdrop-blur-sm">
      <div className="flex items-center gap-2 px-3 py-2">
        <Users size={14} className="text-aware-captain shrink-0" />
        <span className="text-sm font-semibold text-aware-text">{nodeData.label}</span>
      </div>
    </div>
  );
}

export default memo(GroupNode);
