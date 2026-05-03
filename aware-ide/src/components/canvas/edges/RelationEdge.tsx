import { memo } from 'react';
import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from '@xyflow/react';
import type { CanvasEdgeData } from '@/types/canvas';

function RelationEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
  markerEnd,
  style,
}: EdgeProps) {
  const edgeData = data as unknown as CanvasEdgeData | undefined;
  const animated = edgeData?.animated ?? false;
  const label = edgeData?.label ?? '';

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        markerEnd={markerEnd}
        style={{
          stroke: '#6366f1',
          strokeWidth: 2,
          ...style,
          ...(animated ? { strokeDasharray: '6 3' } : {}),
        }}
        className={animated ? 'animate-[dash_1s_linear_infinite]' : ''}
      />
      {label && (
        <EdgeLabelRenderer>
          <div
            className="absolute text-[10px] font-medium text-aware-text bg-aware-panel px-1.5 py-0.5 rounded border border-aware-border pointer-events-all nodrag nopan"
            style={{
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            }}
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

export default memo(RelationEdge);
