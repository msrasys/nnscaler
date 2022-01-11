from cube.graph import IRGraph
from cube.graph.operator import IRFwOperation, IRDataOperation


def PAS(graph: IRGraph, resource):
    """
    Hybrid parallel
    """
    # data operation replication
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, subnode in enumerate(sub_nodes):
                graph.assign(subnode, idx)
    # forward operation
    configs = [
        dict(idx=1, dim=0, num=resource.ngpus),  # linear col
        dict(idx=0, dim=-1, num=resource.ngpus), # gelu col
        dict(idx=0, dim=-1, num=resource.ngpus), # linear row
        dict(idx=0, dim=-1, num=resource.ngpus), # sum
    ]

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    assert len(fnodes) == len(configs)
    for fnode, config in zip(fnodes, configs):
        algo = fnode.algorithms('dim')
        sub_nodes = graph.partition(
            fnode, algo, config=config
        )
        for idx, subnode in enumerate(sub_nodes):
            graph.assign(subnode, idx)
    return graph
