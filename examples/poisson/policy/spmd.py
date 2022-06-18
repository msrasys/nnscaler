from cube.graph import IRGraph
from cube.graph.function import IRConv2D


def PASReplica(graph: IRGraph, resource) -> IRGraph:
    for node in graph.nodes():
        sub_nodes = graph.replicate(node, times=resource.ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    return graph


def PASHaloConv(graph: IRGraph, resource) -> IRGraph:
    for node in graph.nodes():
        if isinstance(node, IRConv2D):
            sub_nodes = list()
            algo = node.algorithms('halo')
            Wnodes = graph.partition(node, algo, idx=0, dim=3, num=resource.ngpus // 2)
            for Wnode in Wnodes:
                algo = Wnode.algorithms('halo')
                Hnodes = graph.partition(Wnode, algo, idx=0, dim=2, num=2)
                sub_nodes += Hnodes
        else:
            sub_nodes = graph.replicate(node, times=resource.ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    return graph
