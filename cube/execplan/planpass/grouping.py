"""
Operation grouping
"""
import os
from typing import List, Dict, Optional, Tuple

from cube.execplan import ExecutionPlan
from cube.execplan.planpass.planpass import PlanPass
from cube.ir.adapter import IRAdapter
from cube.ir.operator import IRBpOperation, IRFwOperation
from cube.ir.cten import IRCell

SCIENTIFIC_COMPUTING = 'SCIENTIFIC_COMPUTING'
_use_new_grouping_algo:Optional[bool] = None

def _set_use_new_grouping_algo(use_new_grouping_algo:Optional[bool]) -> None:
    """
    Set the internal flag whether to use a new grouping algorithm which is faster for grouping forward-only graphs,
    especially for workloads from scientific-computing domains.

    Parameters:
    - use_new_grouping_algo (bool):
        'True' to force the use of the new grouping algorithm.
        'False' to force the use of the old grouping algorithm.
        'None' to use the new grouping algorithm if the environment variable 'SCIENTIFIC_COMPUTING' exists.
    """
    assert use_new_grouping_algo is None or isinstance(use_new_grouping_algo, bool)
    global _use_new_grouping_algo
    _use_new_grouping_algo = use_new_grouping_algo

def _get_use_new_grouping_algo() -> bool:
    if _use_new_grouping_algo is None:
        return SCIENTIFIC_COMPUTING in os.environ
    else:
        return _use_new_grouping_algo
        

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
        fgroups, bgroups = dict(), dict()
        for devid in execplan.devices():
            fgroups[devid], bgroups[devid] = list(), list()
            fpieces, bpieces = list(), list()
            seq = execplan.seq(devid)
            fnodes = []

            def is_forward_node(fnode):
                if isinstance(fnode, IRFwOperation):
                    return True
                if isinstance(fnode, IRAdapter) and fnode.differentiable and fnode.forward:
                    return True
                return False

            for fnode in seq:
                if is_forward_node(fnode):
                    fnodes.append(fnode)
            have_backward = all(fnode.mirror in seq for fnode in fnodes)
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
                if _get_use_new_grouping_algo():

                    for fnode in seq:
                        if is_forward_node(fnode):
                            fpieces.append(fnode)
                        else:
                            if len(fpieces) != 0:
                                fgroups[devid].append(fpieces)
                                bgroups[devid].append(None)

                            # If the fnode is not a "forward node", e.g. it's DataOp node, don't add it into the group.
                            fpieces = []
                            # 'bpieces' is never filled or returned in the inference mode
                    
                    if len(fpieces) != 0:
                        fgroups[devid].append(fpieces)
                        bgroups[devid].append(None)
                
                else: # Not using new algo
                    
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
