from cube.graph import IRGraph
from cube.ir.adapter.adapter import IRAdapter
from cube.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation


def PAS(graph: IRGraph, resource):
    print(graph.extra_repr())
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            sub_nodes = graph.replicate(node, times=resource.ngpus, reset_dependency=False)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    return graph
