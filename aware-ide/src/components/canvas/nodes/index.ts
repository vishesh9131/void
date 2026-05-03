import type { NodeTypes } from '@xyflow/react';
import WorkerNode from './WorkerNode';
import BlockNode from './BlockNode';
import GateNode from './GateNode';
import SandboxNode from './SandboxNode';
import GroupNode from './GroupNode';

export const nodeTypes: NodeTypes = {
  worker: WorkerNode,
  block: BlockNode,
  gate: GateNode,
  sandbox: SandboxNode,
  group: GroupNode,
};
