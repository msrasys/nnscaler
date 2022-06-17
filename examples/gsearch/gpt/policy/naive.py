from cube.graph import IRGraph

def PAS(graph: IRGraph, resource):
    # print(graph.extra_repr())
    for node in graph.nodes():
        graph.assign(node, 0)
    # print(graph.extra_repr())
    return graph