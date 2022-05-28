from cube.graph import IRGraph
from cube.ir.operator import IRFwOperation, IRDataOperation


def PAS(graph: IRGraph, resource):
    """
    Linear Column Partition
    """
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
        if isinstance(node, IRFwOperation):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(
                node, algo, config=dict(idx=1, dim=1, num=resource.ngpus)
            )
            if sub_nodes is None:  # partition fails
                # graph.assign(node, list(range(resource.ngpus)))
                sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    print(graph.extra_repr())
    return graph
