import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { SquareTerminal } from 'lucide-react';
import type { CanvasNodeData } from '@/types/canvas';

function SandboxNode({ data }: NodeProps) {
  const nodeData = data as unknown as CanvasNodeData;
  const output = (nodeData.metadata?.output as string) ?? '';
  const previewLines = output.split('\n').slice(-6).join('\n');

  return (
    <div className="min-w-[240px] max-w-[340px] rounded-lg border border-aware-border bg-aware-bg shadow-lg overflow-hidden">
      <Handle type="target" position={Position.Top} className="!bg-green-500 !w-2.5 !h-2.5" />

      <div className="flex items-center gap-2 px-3 py-1.5 bg-aware-surface border-b border-aware-border">
        <div className="flex gap-1">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500/70" />
          <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/70" />
          <span className="w-2.5 h-2.5 rounded-full bg-green-500/70" />
        </div>
        <SquareTerminal size={13} className="text-green-400 ml-1" />
        <span className="text-xs font-medium text-aware-text truncate">{nodeData.label}</span>
        <span className="ml-auto text-[10px] text-aware-muted capitalize">{nodeData.status}</span>
      </div>

      <div className="p-2">
        <pre className="text-[11px] leading-relaxed text-green-400 font-mono whitespace-pre-wrap break-words min-h-[40px]">
          <code>{previewLines || '$ _'}</code>
        </pre>
      </div>

      <Handle type="source" position={Position.Bottom} className="!bg-green-500 !w-2.5 !h-2.5" />
    </div>
  );
}

export default memo(SandboxNode);
