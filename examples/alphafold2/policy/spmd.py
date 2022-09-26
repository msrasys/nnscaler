from typing import List
from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation

recompute_info = {
    'MSAAttention': True,
    'MSAAttentionWithBias': True,
    'MSARowAttentionWithPairBias': True,
    'MSATransition': True,
    'OuterProductMean': True,
    'TriangleMultiplication': True,
    'TriangleAttentionNode': True,
    'PairTransition': True,
    'add': False,
    'sum': False,
    'layernorm': False,
    'transpose': False,
}

def PASData(graph: IRGraph, resource):
    devs = list(range(resource.ngpus))

    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, num=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
            batch_dim = node.get_batch_dims()[0]

    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            if node.name == 'mul':
                sub_nodes = graph.replicate(node, times=resource.ngpus)
                for devid, sub_node in zip(devs, sub_nodes):
                    graph.assign(sub_node, devid)
                continue
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(node,
                                        algo,
                                        idx=0,
                                        dim=batch_dim,
                                        num=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
            if node.name in recompute_info and recompute_info[node.name] == True:
                graph.recompute(sub_nodes)
    return graph
