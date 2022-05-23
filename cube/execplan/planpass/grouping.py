"""
Operation grouping
"""
from typing import List, Dict, Tuple

from cube.execplan import ExectuionPlan
from cube.execplan.planpass.planpass import PlanPass
from cube.graph.adapter.adapter import IRAdapter
from cube.graph.operator.operator import IRBpOperation, IRFwOperation
from cube.ir.cten import IRCell


class Grouping(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan) -> ExectuionPlan:
        """
        Group contiguous forward and contiguous backward
        into subgraph
        """
        graph = execplan.graph
        fgroups, bgroups = Grouping.group(execplan)
        for devid in execplan.devices():
            for fpieces, bpieces in zip(fgroups[devid], bgroups[devid]):
                fsubgraph = graph.subgraph(fpieces)
                fsubgraph.device = devid
                if bpieces is not None:
                    bsubgraph = graph.subgraph(bpieces)
                    bsubgraph.device = devid
                    IRCell.make_pair(fsubgraph, bsubgraph)
                subgraphs = [fsubgraph] if bpieces is None else [fsubgraph, bsubgraph]
                for subgraph in subgraphs:
                    pieces = subgraph.nodes()
                    # update graph: replace the nodes with the subgraph
                    idx = graph.nodes().index(pieces[0])
                    graph._nodes.insert(idx, subgraph)
                    for node in pieces:
                        graph._nodes.remove(node)
                    # update execution plan: replace the nodes with the subgraph
                    idx = execplan.sequence(devid).index(pieces[0])
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
        fgroups, bgroups = dict(), dict()
        for devid in execplan.devices():
            fgroups[devid], bgroups[devid] = list(), list()
            fpieces, bpieces = list(), list()
            seq = execplan.sequence(devid)
            fnodes = [fnode for fnode in seq if isinstance(fnode, IRFwOperation)]
            have_backward = all([fnode.mirror in seq for fnode in fnodes])
            # training
            if have_backward:
                bnodes = [fnode.mirror for fnode in fnodes]
                for fnode, bnode in zip(fnodes + [-1], bnodes + [-1]):
                    fconsecutive = Grouping.consecutive(seq, fpieces, fnode)
                    bconsecutive = Grouping.consecutive(seq, bpieces, bnode)
                    if fconsecutive and bconsecutive:
                        fpieces.append(fnode)
                        bpieces.insert(0, bnode)
                    else:
                        if len(fpieces) != 0:
                            fgroups[devid].append(fpieces)
                            bgroups[devid].append(bpieces)
                        fpieces, bpieces = [fnode], [bnode]
            # inference
            else:
                for fnode in fnodes + [-1]:
                    fconsecutive = Grouping.consecutive(seq, fpieces, fnode)
                    if fconsecutive:
                        fpieces.append(fnode)
                        bpieces.append(None)
                    else:
                        if len(fpieces) != 0:
                            fgroups[devid].append(fpieces)
                            bgroups[devid].append(None)
                        fpieces, bpieces = [fnode], [None]
        return fgroups, bgroups

    @staticmethod
    def consecutive(seq: List[IRCell], pieces: List[IRCell], node: IRCell):
        """
        Check whether the piecies with new node
        is consecutive in the sequence.

        Assume all the node in pieces will apear in seq.
        If node not in the sequence, will return False.
        """
        if len(pieces) == 0:
            return True
        if node not in seq:
            return False
        idx = seq.index(node)
        pidx = [seq.index(pnode) for pnode in pieces]
        # check whether pieces is consecutive
        if max(pidx) - min(pidx) != len(pidx) - 1:
            return False
        # check whether new node adding new node is consecutive
        if idx != max(pidx) + 1 and idx != min(pidx) - 1:
            return False
        return True


class GroupingAdapter(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan) -> ExectuionPlan:
        for devid in execplan.devices():
            groups: List[List[IRAdapter]] = GroupingAdapter.consecutive(
                execplan.sequence(devid))
            for adapters in groups:
                if len(adapters) <= 1:
                    continue
                sprims, tprims, mprims = list(), list(), list()
                inputs, idevices = list(), list()
                outputs, odevices = list(), list()
                for adapter in adapters:
                    sprims += adapter.prims(move=False, merge=False, coll=False)
                    tprims += adapter.prims(select=False, merge=False)
                    mprims += adapter.prims(select=False, move=False, coll=False)
                    for idx, input in enumerate(adapter.inputs()):
                        if devid in adapter.idevice(idx):
                            if input not in inputs:
                                inputs.append(input)
                                idevices.append(adapter.idevice(idx))
                    for idx, output in enumerate(adapter.outputs()):
                        if devid in adapter.odevice(idx):
                            if output not in outputs:
                                outputs.append(output)
                                odevices.append(adapter.odevice(idx))
                prims = sprims + tprims + mprims
                fused_adapter = IRAdapter(prims,
                    inputs = inputs, idevices = idevices,
                    outputs = outputs, odevices = odevices)
                start = execplan.sequence(devid).index(adapters[0])
                end = execplan.sequence(devid).index(adapters[-1])
                for _ in range(end - start + 1):
                    execplan.at(devid).pop(start)
                execplan.at(devid).insert(start, fused_adapter)
        return execplan

    @staticmethod
    def consecutive(seq: List[IRCell]) -> List[List[IRAdapter]]:
        group = list()
        curr = list()
        curr_idx = -1
        for idx, node in enumerate(seq + [None]):
            if isinstance(node, IRAdapter) and idx == curr_idx + 1:
                curr.append(node)
            else:
                if len(curr) != 0:
                    group.append(curr)
                curr = list()
            curr_idx = idx
        return group
