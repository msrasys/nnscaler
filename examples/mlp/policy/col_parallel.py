from cube.graph import IRGraph
from cube.graph.operator.operator import IRDataOperation, IRFwOperation


def P(graph, resource):
    """
    P policy
    """
    for node in graph.nodes():
        if isinstance(node, IRFwOperation) or isinstance(node, IRDataOperation):
            algo = node.algorithms('column')
            if algo:
                sub_nodes = graph.partition(
                    node, algo, config=dict(chunk_num=resource.ngpus)
                )
            else:
                # graph.assign(node, list(range(resource.ngpus)))
                sub_nodes = graph.replicate(node, times=resource.ngpus)
        # device hint
        for idx, node in enumerate(sub_nodes):
                node.tag = idx
    return graph


def A(graph, resource):
    """
    A policy
    """
    for node in graph.nodes():
        if node.tag is not None:
            device = node.tag
            graph.assign(node, device)
    return graph


def S(graph, resource):
    """
    Schedule Policy. => use default schedule
    """
    return graph


def PAS(graph: IRGraph, resource):
    """
    Linear Column Partition
    """
    for node in graph.nodes():
        if isinstance(node, IRFwOperation) or isinstance(node, IRDataOperation):
            algo = node.algorithms('column')
            if algo:
                sub_nodes = graph.partition(
                    node, algo, config=dict(chunk_num=resource.ngpus)
                )
            else:
                # graph.assign(node, list(range(resource.ngpus)))
                sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    print(graph.extra_repr())
    return graph
