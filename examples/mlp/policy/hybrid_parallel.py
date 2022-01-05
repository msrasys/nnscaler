from cube.graph import IRGraph
from cube.graph.operator.operator import IRDataOperation, IRFwOperation


def PAS(graph: IRGraph, resource):
    """
    Linear Hybrid Partition
    """
    for idx, node in enumerate(graph.nodes()):
        if isinstance(node, IRFwOperation) or isinstance(node, IRDataOperation):
            algo = node.algorithms('dim')
            if algo:
                sub_nodes = graph.partition(
                    node, algo,
                    config=dict(idx=1, dim=(idx+1)%2, num=resource.ngpus)
                )
            else:
                sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    print(graph.extra_repr())
    return graph
