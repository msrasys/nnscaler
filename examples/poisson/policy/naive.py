from cube.graph import IRGraph
from cube.graph.operator.function import IRConv2D

def PAS(graph: IRGraph, resource):
    for node in graph.nodes():
        if isinstance(node, IRConv2D):
            algo = node.algorithms('halo')
            sub_nodes = graph.partition(node, algo, config=dict(idx=0, dim=3, num=resource.ngpus))
        else:
            sub_nodes = graph.replicate(node, times=resource.ngpus)
        # sub_nodes = graph.replicate(node, times=resource.ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    print(graph.extra_repr())
    return graph
