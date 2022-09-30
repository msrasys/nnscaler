from typing import List

from numpy import TooHardError
from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation, IRBpOperation

recompute_info = {
    'MSAAttention': True,
    'MSAAttentionWithBias': True,
    'MSARowAttentionWithPairBias': True,
    'MSAColAttention': True,
    'MSATransition': True,
    'OuterProductMean': True,
    'TriangleMultiplicationOut': True,
    'TriangleMultiplicationIn': True,
    'TriangleAttentionNodeStart': True,
    'TriangleAttentionNodeEnd': True,
    'PairTransition': True,
    'add': False,
    'sum': False,
    'layernorm': False,
    'transpose': False,
}


def PASSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)
    return graph


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
            if node.name in recompute_info and recompute_info[
                    node.name] == True:
                graph.recompute(sub_nodes)
    return graph


def PASMegatron(graph: IRGraph, resource):
    tp_size = resource.ngpus
    tp_devs = list(range(tp_size))

    def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int], idx: int,
            dim: int):
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(node,
                                    algo,
                                    idx=idx,
                                    dim=dim,
                                    num=len(devs))
        assert sub_nodes is not None
        for devid, sub_node in zip(devs, sub_nodes):
            graph.assign(sub_node, devid)
        return sub_nodes

    def _replica(graph: IRGraph, node: IRFwOperation, devs: List[int]):
        sub_nodes = graph.replicate(node, times=len(devs))
        for dev_id, sub_node in zip(devs, sub_nodes):
            graph.assign(sub_node, dev_id)
        return sub_nodes

    pred_name = ''
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            # _tp(graph, node, tp_devs, 0, 1)
            _replica(graph, node, tp_devs)
        elif isinstance(node, IRFwOperation):
            if node.name == 'MSARowAttentionWithPairBias':
                _tp(graph, node, tp_devs, 0, 1)
                pred_name = node.name
            elif node.name == 'MSAColAttention':
                _tp(graph, node, tp_devs, 0, 2)
                pred_name = node.name
            elif node.name == 'MSATransition':
                _tp(graph, node, tp_devs, 0, 2)
                pred_name = node.name
            elif node.name == 'OuterProductMean':
                _tp(graph, node, tp_devs, 0, 2)
                pred_name = node.name
            elif node.name == 'TriangleMultiplicationOut':
                _tp(graph, node, tp_devs, 0, 1)
                pred_name = node.name
            elif node.name == 'TriangleMultiplicationIn':
                _tp(graph, node, tp_devs, 0, 2)
                pred_name = node.name
            elif node.name == 'TriangleAttentionNodeStart':
                _tp(graph, node, tp_devs, 0, 1)
                pred_name = node.name
            elif node.name == 'TriangleAttentionNodeEnd':
                _tp(graph, node, tp_devs, 0, 2)
                pred_name = node.name
            elif node.name == 'PairTransition':
                _tp(graph, node, tp_devs, 0, 1)
                pred_name = node.name
            else:
                if node.name == 'add':
                    if pred_name == 'PairTransition':
                        _tp(graph, node, tp_devs, 0, 1)
                    elif pred_name == 'TriangleAttentionNodeEnd':
                        _tp(graph, node, tp_devs, 0, 2)
                    elif pred_name == 'TriangleAttentionNodeStart':
                        _tp(graph, node, tp_devs, 0, 1)
                    elif pred_name == 'TriangleMultiplicationIn':
                        _tp(graph, node, tp_devs, 0, 2)
                    elif pred_name == 'TriangleMultiplicationOut':
                        _tp(graph, node, tp_devs, 0, 1)
                    elif pred_name == 'OuterProductMean':
                        _tp(graph, node, tp_devs, 0, 1)
                    elif pred_name == 'MSATransition':
                        _tp(graph, node, tp_devs, 0, 2)
                    elif pred_name == 'MSAColAttention':
                        _tp(graph, node, tp_devs, 0, 2)
                    elif pred_name == 'MSARowAttentionWithPairBias':
                        _tp(graph, node, tp_devs, 0, 1)
                    else:
                        assert False
                elif node.name == 'layernorm':
                    if pred_name == 'TriangleAttentionNodeEnd':
                        _tp(graph, node, tp_devs, 0, 1)
                    elif pred_name == 'TriangleAttentionNodeStart':
                        _tp(graph, node, tp_devs, 0, 2)
                    elif pred_name == 'TriangleMultiplicationIn':
                        _tp(graph, node, tp_devs, 0, 1)
                    elif pred_name == 'MSATransition':
                        _tp(graph, node, tp_devs, 0, 2)
                    elif pred_name == 'MSAColAttention':
                        _tp(graph, node, tp_devs, 0, 2)
                    elif pred_name == 'MSARowAttentionWithPairBias':
                        _tp(graph, node, tp_devs, 0, 2)
                    elif pred_name == '':
                        _tp(graph, node, tp_devs, 0, 1)
                    else:
                        assert False
                else:
                    print('replica node:', node.name)
                    _replica(graph, node, tp_devs)
    return graph
