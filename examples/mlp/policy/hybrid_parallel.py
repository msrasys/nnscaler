from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation


def PAS(graph: IRGraph, resource):
    """
    Linear Hybrid Partition
    """
    for idx, node in enumerate(graph.nodes()):
        if isinstance(node, IRDataOperation):
            sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
        if isinstance(node, IRFwOperation):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(
                node, algo, config=dict(idx=1, dim=(idx+1)%2, num=resource.ngpus)
            )
            if sub_nodes is None:  # partition fails
                sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    print(graph.extra_repr())
    return graph
