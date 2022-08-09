from typing import List

from cube.graph import IRGraph
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation


def PASSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1
    # print(graph.extra_repr())
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)
    return graph