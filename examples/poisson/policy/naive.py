from cube.graph import IRGraph

def PAS(graph: IRGraph, resource):
    for node in graph.nodes():
        sub_nodes = graph.replicate(node, times=resource.ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    print(graph.extra_repr())
    return graph
