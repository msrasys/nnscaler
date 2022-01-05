from cube.graph import IRGraph
from cube.graph.operator.operator import IRFwOperation, IRDataOperation


def PAS(graph: IRGraph, resource):
    """
    Linear Column Partition
    """
    for node in graph.nodes():
        if isinstance(node, IRFwOperation) or isinstance(node, IRDataOperation):
            algo = node.algorithms('dim')
            if algo:
                sub_nodes = graph.partition(
                    node, algo, config=dict(idx=1, dim=1, num=resource.ngpus)
                )
            else:
                sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    print(graph.extra_repr())
    return graph
