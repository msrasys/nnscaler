from typing import List
from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation

recompute_info = {
    'MSAAttention': True,
    'MSAAttentionWithBias': True,
    'MSARowAttentionWithPairBias': True,
    'MSAColAttention': True,
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

def PASMegatron(graph: IRGraph, resource):
    tp_size = resource.ngpus
    tp_devs = list(range(tp_size))

    def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int], idx: int, dim: int):
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
        assert sub_nodes is not None
        for devid, sub_node in zip(devs, sub_nodes):
            graph.assign(sub_node, devid)
        return sub_nodes

    def _replica(graph: IRGraph, node: IRFwOperation, devs: List[int]):
        sub_nodes = graph.replicate(node, times=len(devs))
        for dev_id, sub_node in zip(devs, sub_nodes):
            graph.assign(sub_node, dev_id)
        return sub_nodes

    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, num=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
            batch_dim = node.get_batch_dims()[0]