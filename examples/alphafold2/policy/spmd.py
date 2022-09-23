from typing import List
from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation

def PASData(graph: IRGraph, resource):

    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, num=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
            batch_dim = node.get_batch_dims()[0]

    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(node,
                                        algo,
                                        idx=0,
                                        dim=batch_dim,
                                        num=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
    return graph
