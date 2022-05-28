from cube.graph import IRGraph
from cube.ir.adapter.adapter import IRAdapter
from cube.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation


def PAS(graph: IRGraph, resource):
    print(graph.extra_repr())
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)
    return graph