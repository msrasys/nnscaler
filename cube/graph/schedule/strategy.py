from typing import Tuple, Dict, Any, List
from cube.graph.graph import IRGraph, IRSegment
from cube.ir.adapter.adapter import IRAdapter
from cube.ir.cten import IRCell
from cube.ir.operator import IRFwOperation


class IRScheduleStrategy:

    def __init__(self, graph: IRGraph, nmicros: int, devmesh: Tuple[Tuple[int]]) -> None:
        self.graph : IRGraph = graph
        self.nmicros : int = nmicros
        self.devmesh: Tuple[Tuple[int]] = devmesh
        self.inner_groups: List[IRSegment] = [None] * len(devmesh)
        self.cross_groups: List[IRAdapter] = [None] * len(devmesh)
        self.signature: str = ''

    def apply(self, graph: IRGraph) -> IRGraph:
        raise NotImplementedError

    def kwargs(self, device: int) -> Dict[str, Any]:
        raise NotImplementedError

    def segmentation(self):
        """!
        Group operators into segments corresponding to devmesh.

        A greedy grouping is applied for each group given the device mesh.
        The non-differentiable adapters need to be moved at the boundary
        of device mesh, as the cross group communication.
        """
        def differientiable(node: IRCell) -> bool:
            return isinstance(node, IRFwOperation) or \
                   (isinstance(node, IRAdapter) and node.forward and node.differentiable)
        
        devmesh = self.devmesh
        inner_groups: List[List[IRCell]] = [[] for _ in range(len(devmesh))]
        cross_groups: List[List[IRAdapter]] = [[] for _ in range(len(devmesh))]
        sid = 0
        for node in self.graph.nodes():
            if not (isinstance(node, (IRFwOperation, IRAdapter))):
                continue
            devs = set(node.device)
            if differientiable(node):
                while sid < len(devmesh) and not devs.issubset(devmesh[sid]):
                    sid += 1
                assert sid < len(devmesh), f"invalid stategy with graph placement" 
                inner_groups[sid].append(node)
            else:
                if not (isinstance(node, IRAdapter) and node.forward):
                    continue
                assert not devs.issubset(devmesh[sid]), f"find a non-differentiable adapter in devmesh: {devmesh[sid]}"
                cross_mesh = devmesh[sid] + devmesh[sid+1] if sid < len(devmesh) - 1 else devmesh[sid]
                assert devs.issubset(set(cross_mesh))
                cross_groups[sid].append(node)

        # move non-differentiable adapter to the boundary of groups
        for igroup, cgroup in zip(inner_groups, cross_groups):
            if len(igroup) == 0:
                print('warning: find a group with no operator')
                continue
            last_node: IRCell = igroup[-1]
            for fadapter in cgroup[::-1]:
                success = self.graph.schedule(fadapter, 'after', last_node)
                if fadapter.mirror is not None and last_node.mirror is not None:
                    success = self.graph.schedule(
                        fadapter.mirror, 'before', last_node.mirror
                    )
                if not success:
                    raise RuntimeError("Fail to schedule non-differentiable adapter to group boundaries")

        # grouping
        for gid in range(len(devmesh)):
            # group computation groups
            igroup = inner_groups[gid]
            if len(igroup) != 0:
                fsegment = self.graph.group(igroup)
                bnodes = [n.mirror for n in igroup[::-1] if n.mirror is not None]
                if len(bnodes) != 0:
                    bsegment = self.graph.group(bnodes)
                    IRCell.make_pair(fsegment, bsegment)
                self.inner_groups[gid] = fsegment
            else:
                self.inner_groups[gid] = None
            # merge cross communication adapters
            cgroup = cross_groups[gid]
            if len(cgroup) == 1:
                self.cross_groups[gid] = cgroup[0]
            elif len(cgroup) > 1:
                fadapter = IRAdapter.merge(cgroup)
                bnodes = [n.mirror for n in igroup[::-1] if n.mirror is not None]
                if len(bnodes) != 0:
                    badapter = IRAdapter.merge(bnodes)
                    IRCell.make_pair(fadapter, badapter)
                self.cross_groups[gid] = fadapter
            else:
                self.cross_groups[gid] = None
