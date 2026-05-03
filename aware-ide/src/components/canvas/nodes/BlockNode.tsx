import { memo } from 'react';
import { Handle, Position, type NodeProps } from '@xyflow/react';
import { Code } from 'lucide-react';
import type { CanvasNodeData } from '@/types/canvas';

const MAX_PREVIEW_LINES = 5;

function BlockNode({ data }: NodeProps) {
  const nodeData = data as unknown as CanvasNodeData;
  const content = nodeData.content ?? '';
  const lines = content.split('\n');
  const preview = lines.slice(0, MAX_PREVIEW_LINES).join('\n');
  const truncated = lines.length > MAX_PREVIEW_LINES;

  return (
    <div className="min-w-[220px] max-w-[320px] rounded-lg border border-aware-border bg-aware-surface shadow-lg">
      <Handle type="target" position={Position.Top} className="!bg-aware-accent !w-2.5 !h-2.5" />

      <div className="flex items-center gap-2 px-3 py-2 border-b border-aware-border">
        <Code size={14} className="text-aware-accent shrink-0" />
        <span className="text-sm font-medium text-aware-text truncate">{nodeData.label}</span>
      </div>

      {content && (
        <div className="px-3 py-2">
          <pre className="text-[11px] leading-relaxed text-aware-muted font-mono whitespace-pre-wrap break-words bg-aware-bg rounded p-2 overflow-hidden">
            <code>{preview}</code>
          </pre>
          {truncated && (
            <p className="text-[10px] text-aware-muted mt-1">
              +{lines.length - MAX_PREVIEW_LINES} more lines
            </p>
          )}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="!bg-aware-accent !w-2.5 !h-2.5" />
    </div>
  );
}

export default memo(BlockNode);
