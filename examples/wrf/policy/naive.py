from cube.graph import IRGraph
from cube.graph.function import IRConv2D

def PAS(graph: IRGraph, resource):
    for node in graph.nodes():
        graph.assign(node, 0)
    print(graph.extra_repr())
    return graph
