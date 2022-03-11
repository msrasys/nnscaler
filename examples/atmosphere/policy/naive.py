from cube.graph import IRGraph
from cube.graph.adapter.adapter import IRAdapter
from cube.graph.operator.operator import IRBpOperation, IRDataOperation, IRFwOperation


def PAS(graph: IRGraph, resource):
    print(graph.extra_repr())
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)
    return graph
