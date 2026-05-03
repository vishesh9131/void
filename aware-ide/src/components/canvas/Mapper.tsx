import { useCallback } from 'react';
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  Panel,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { useCanvasStore } from '@/stores/canvasStore';
import { nodeTypes } from './nodes';
import { edgeTypes } from './edges';
import NodeToolbar from './NodeToolbar';

export default function Mapper() {
  const nodes = useCanvasStore((s) => s.nodes);
  const edges = useCanvasStore((s) => s.edges);
  const onNodesChange = useCanvasStore((s) => s.onNodesChange);
  const onEdgesChange = useCanvasStore((s) => s.onEdgesChange);
  const onConnect = useCanvasStore((s) => s.onConnect);

  const proOptions = { hideAttribution: true };

  const minimapNodeColor = useCallback(() => '#6366f1', []);

  return (
    <div className="w-full h-full bg-aware-bg">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        defaultEdgeOptions={{ type: 'relation' }}
        fitView
        proOptions={proOptions}
        className="bg-aware-bg"
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#2a2a2a" />

        <Controls
          showInteractive={false}
          className="!bg-aware-panel !border-aware-border !rounded-lg !shadow-lg [&>button]:!bg-aware-panel [&>button]:!border-aware-border [&>button]:!text-aware-muted [&>button:hover]:!bg-aware-hover [&>button]:!fill-aware-muted"
        />

        <MiniMap
          nodeColor={minimapNodeColor}
          maskColor="rgba(13, 13, 13, 0.85)"
          className="!bg-aware-surface !border-aware-border !rounded-lg"
          pannable
          zoomable
        />

        <Panel position="top-center" className="!m-2">
          <NodeToolbar />
        </Panel>

        <Panel position="bottom-left" className="!m-2">
          <div className="bg-aware-panel border border-aware-border rounded-md px-3 py-1.5 text-xs text-aware-muted shadow-lg">
            {nodes.length} node{nodes.length !== 1 ? 's' : ''} / {edges.length} edge{edges.length !== 1 ? 's' : ''}
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
}
