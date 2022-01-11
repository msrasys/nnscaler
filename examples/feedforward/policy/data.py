from cube.graph import IRGraph
from cube.graph.operator import IRFwOperation, IRDataOperation


def PAS(graph: IRGraph, resource):
    """
    Data Parallel
    """
    # data operation
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
                node, algo, config=dict(idx=0, dim=batch_dim, num=resource.ngpus)
            )
            for idx, subnode in enumerate(sub_nodes):
                graph.assign(subnode, idx)
    return graph
