from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation


def PAS(graph: IRGraph, resource):
    """
    Linear Column Partition
    """
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, config=dict(num=resource.ngpus))
            for idx, subnode in enumerate(sub_nodes):
                graph.assign(subnode, idx)
            batch_dim = node.get_batch_dims()[0]
    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(
                node, algo, config=dict(idx=0, dim=batch_dim, num=resource.ngpus))
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    print(graph.extra_repr())
    return graph
