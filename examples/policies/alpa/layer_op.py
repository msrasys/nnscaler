from typing import List, Dict, Tuple
import more_itertools

from cube.ir.cten import IRCell
from cube.ir.operator import IRFwOperation
from cube.graph.graph import IRGraph
from cube.graph.function.anchor import IRGraphAnchor


class IRLayerOp(IRCell):

    def __init__(self, nodes: List[IRCell], layer_id: int = None):
        super().__init__('layer_op', 'layer_op', 0, 0, init_outputs=False)
        self.nodes = nodes
        self.layer_id : int = layer_id


def cluster_to_layer_ops(nodes: List[IRFwOperation]) -> List[IRLayerOp]:
    layer_ops: List[IRLayerOp] = []
    ops = []
    for node in nodes:
        if isinstance(node, IRGraphAnchor):
            if len(ops) != 0:
                layer_ops.append(IRLayerOp(ops, layer_id=len(layer_ops)))
            ops = [node]
        elif isinstance(node, IRFwOperation):
            ops.append(node)
    if len(ops) != 0:
        layer_ops.append(IRLayerOp(ops, layer_id=len(layer_ops)))
    return layer_ops


def annotate_structure(graph: IRGraph) -> List[Tuple[IRFwOperation]]:
    """Annotate graph stucture in generated code"""
    anchors = graph.select(ntype=IRGraphAnchor)
    for idx, anchor in enumerate(anchors):
        nidx = graph.index(anchor)
        graph.node(nidx + 1).comment = f'===> split position {idx}: {anchor.name}'
    fnodes = graph.select(ntype=IRFwOperation)
    subgraphs = more_itertools.split_before(fnodes, lambda n: isinstance(n, IRGraphAnchor))
    return list(subgraphs)
    