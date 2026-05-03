import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Shield } from 'lucide-react';
import type { CanvasNodeData } from '@/types/canvas';

const statusStyles: Record<string, string> = {
  idle: 'border-gray-600 text-gray-400',
  running: 'border-yellow-500 text-yellow-400',
  done: 'border-green-500 text-green-400',
  error: 'border-red-500 text-red-400',
};

function GateNode({ data }: NodeProps) {
  const nodeData = data as unknown as CanvasNodeData;
  const style = statusStyles[nodeData.status] ?? statusStyles.idle;

  return (
    <div className="flex items-center justify-center">
      <Handle type="target" position={Position.Top} className="!bg-aware-warn !w-2.5 !h-2.5" />

      <div
        className={`w-[120px] h-[120px] border-2 bg-aware-surface shadow-lg flex flex-col items-center justify-center gap-1.5 ${style}`}
        style={{ clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)' }}
      >
        <Shield size={16} className="shrink-0" />
        <span className="text-xs font-medium text-aware-text text-center px-4 truncate max-w-full">
          {nodeData.label}
        </span>
        <span className="text-[10px] text-aware-muted capitalize">{nodeData.status}</span>
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-aware-warn !w-2.5 !h-2.5" />
    </div>
  );
}

export default memo(GateNode);
