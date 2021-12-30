from cube.graph import IRGraph
from cube.graph.operator.operator import IRDataOperation, IRFwOperation


def PAS(graph: IRGraph, resource):
    """
    Linear Hybrid Partition
    """
    for idx, node in enumerate(graph.nodes()):
        if isinstance(node, IRFwOperation) or isinstance(node, IRDataOperation):
            if idx % 2 == 0:
                algo = node.algorithms('row')
            else:
                algo = node.algorithms('column')
            if algo:
                sub_nodes = graph.partition(
                    node, algo, config=dict(chunk_num=resource.ngpus)
                )
            else:
                sub_nodes = [node]
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    print(graph.extra_repr())
    return graph
