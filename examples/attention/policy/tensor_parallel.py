from cube.graph import IRGraph
from cube.graph.operator.operator import IRDataOperation, IRFwOperation


def PAS(graph: IRGraph, resource):
    # data loader
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            graph.assign(node, list(range(resource.ngpus)))
    fnodes = [isinstance(node, IRFwOperation) for node in graph.nodes()]
    for idx, node in enumerate(fnodes):
        if idx == 0:
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(
                node, algo, config=dict(idx=0, dim=1, num=resource.ngpus)
            )
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
    print(graph.extra_repr())
    return graph
        
