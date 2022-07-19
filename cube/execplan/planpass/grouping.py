"""
Operation grouping
"""
from typing import List, Dict, Tuple

from cube.execplan import ExecutionPlan
from cube.execplan.planpass.planpass import PlanPass
from cube.ir.adapter import IRAdapter
from cube.ir.operator import IRFwOperation
from cube.ir.cten import IRCell
        

class Grouping(PlanPass):
    @staticmethod
    def apply(execplan: ExecutionPlan) -> ExecutionPlan:
        """
        Group contiguous differentiable operators segments
        """
        graph = execplan.graph
        fgroups, bgroups = Grouping.group(execplan)
        for devid in execplan.devices():
            for fpieces, bpieces in zip(fgroups[devid], bgroups[devid]):
                fsubgraph = graph.segment(fpieces)
                if bpieces is not None:
                    bsubgraph = graph.segment(bpieces)
                    IRCell.make_pair(fsubgraph, bsubgraph)
                subgraphs = [fsubgraph] if bpieces is None else [fsubgraph, bsubgraph]
                for subgraph in subgraphs:
                    # update execution plan: replace the nodes with the subgraph
                    pieces = subgraph.nodes()
                    idx = execplan.seq(devid).index(pieces[0])
                    execplan.at(devid).insert(idx, subgraph)
                    for node in pieces:
                        execplan.at(devid).remove(node)
        return execplan

    @staticmethod
    def group(execplan) -> Tuple[Dict[int, List[List[IRCell]]],]:
        """
        Return forward groups and corresponding
        backward groups for each device.

        Each group can be indexed by device id.
        Each device id contains a list of forward / backward operations

        Returns:
            Tuple: (fgroups, bgroups)
        """
        def is_forward_node(fnode):
            if isinstance(fnode, IRFwOperation):
                return True
            if isinstance(fnode, IRAdapter) and fnode.differentiable and fnode.forward:
                return True
            return False

        fgroups, bgroups = dict(), dict()
        for devid in execplan.devices():
            fgroups[devid], bgroups[devid] = list(), list()
            seq = execplan.seq(devid)
            fnodes = [node for node in seq if is_forward_node(node)]
            have_backward = all(fnode.mirror in seq for fnode in fnodes)
            fpieces = []

            for fnode in seq:
                if is_forward_node(fnode):
                    fpieces.append(fnode)
                else:
                    if len(fpieces) != 0:
                        fgroups[devid].append(fpieces)
                    fpieces = []

            if len(fpieces) != 0:
                fgroups[devid].append(fpieces)

            for pieces in fgroups[devid]:
                if have_backward:
                    bpieces = [fnode.mirror for fnode in pieces[::-1] if fnode.mirror is not None]
                    bgroups[devid].append(bpieces)
                else:
                    bgroups[devid].append(None)

        return fgroups, bgroups
