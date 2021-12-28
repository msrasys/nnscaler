"""
Operation grouping
"""

from typing import List, Dict

from cube.execplan import ExectuionPlan
from cube.execplan.planpass.planpass import PlanPass
from cube.graph.operator.operator import IRBpOperation, IRFwOperation
from cube.graph.adapter.adapter import IRAdapter
from cube.ir.cten import IRCell


class Grouping(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan) -> ExectuionPlan:
        """
        Group contiguous forward and contiguous backward
        into subgraph
        """
        graph = execplan.graph
        # step 1: group forward + adapter
        groups = Grouping.group(execplan, [IRFwOperation])
        for devid in execplan.devices():
            for pieces in groups[devid]:
                subgraph = graph.subgraph(pieces)
                subgraph.device = devid
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
        # step 2: group backward
        groups = Grouping.group(execplan, [IRBpOperation])
        for devid in execplan.devices():
            for pieces in groups[devid]:
                subgraph = graph.subgraph(pieces)
                subgraph.device = devid
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
    def group(execplan, node_types: List) -> Dict[int, List[List[IRCell]]]:
        groups = dict()
        for devid in execplan.devices():
            groups[devid] = list()
            pieces = list()
            dev_seq = execplan.sequence(devid) + [None]
            for node in dev_seq:
                if all([isinstance(node, ntype) for ntype in node_types]):
                    pieces.append(node)
                else:
                    if len(pieces) != 0:
                        groups[devid].append(pieces)
                    pieces = list()
        return groups
