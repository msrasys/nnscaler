from cube.graph import IRGraph
from cube.graph.operator.operator import IRDataOperation, IRFwOperation


def PAS(graph: IRGraph, resource):
    print(graph.extra_repr())
    return graph
