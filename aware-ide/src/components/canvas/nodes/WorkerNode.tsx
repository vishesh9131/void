import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Bot } from 'lucide-react';
import type { CanvasNodeData } from '@/types/canvas';

const statusColors: Record<string, string> = {
  idle: 'bg-gray-500',
  thinking: 'bg-yellow-400',
  working: 'bg-blue-500',
  done: 'bg-green-500',
  error: 'bg-red-500',
};

const statusBorders: Record<string, string> = {
  idle: 'border-gray-600',
  thinking: 'border-yellow-500',
  working: 'border-blue-500',
  done: 'border-green-500',
  error: 'border-red-500',
};

function WorkerNode({ data }: NodeProps) {
  const nodeData = data as unknown as CanvasNodeData;
  const status = (nodeData.metadata?.agentStatus as string) ?? 'idle';
  const currentTask = nodeData.metadata?.currentTask as string | undefined;
  const progress = (nodeData.metadata?.progress as number) ?? 0;

  return (
    <div
      className={`min-w-[200px] rounded-lg border-2 bg-aware-surface shadow-lg ${statusBorders[status] ?? 'border-aware-border'}`}
    >
      <Handle type="target" position={Position.Top} className="!bg-aware-worker !w-2.5 !h-2.5" />

      <div className="flex items-center gap-2 px-3 py-2 border-b border-aware-border">
        <Bot size={14} className="text-aware-worker shrink-0" />
        <span className="text-sm font-medium text-aware-text truncate">{nodeData.label}</span>
        <span className={`ml-auto w-2.5 h-2.5 rounded-full shrink-0 ${statusColors[status] ?? 'bg-gray-500'}`} />
      </div>

      <div className="px-3 py-2 space-y-1.5">
        <p className="text-xs text-aware-muted capitalize">{status}</p>

        {currentTask && (
          <p className="text-xs text-aware-text truncate" title={currentTask}>
            {currentTask}
          </p>
        )}

        {status === 'working' && (
          <div className="w-full h-1.5 rounded-full bg-aware-bg overflow-hidden">
            <div
              className="h-full rounded-full bg-blue-500 transition-all duration-300"
              style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            />
          </div>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-aware-worker !w-2.5 !h-2.5" />
    </div>
  );
}

export default memo(WorkerNode);
