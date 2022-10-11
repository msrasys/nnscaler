from typing import List

from numpy import TooHardError
from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation, IRBpOperation

recompute_info = {
    'MSARowAttentionWithPairBias': True,
    'MSAColAttention': True,
    'MSATransition': True,
    'OPMLeftProj': True,
    'OPMRightProj': True,
    'OuterProductMean': True,
    'TMOLeftProj': True,
    'TMORightProj': True,
    'TMOGate': True,
    'TriangleMultiplicationOut': True,
    'TMILeftProj': True,
    'TMIRightProj': True,
    'TMIGate': True,
    'TriangleMultiplicationIn': True,
    'TANSBias': True,
    'TriangleAttentionNodeStart': True,
    'TANEBias': True,
    'TriangleAttentionNodeEnd': True,
    'PairTransition': True,
    'add': False,
    'sum': False,
    'layernorm': False,
    'multi2ref': False,
}


# coshard
def _coshard(graph: IRGraph, node: IRFwOperation, devs: List[int],
             colocate: int, idx: int, dim: int):
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node,
                                algo,
                                idx=idx,
                                dim=dim,
                                num=colocate * len(devs))
    assert sub_nodes is not None
    graph.recompute(sub_nodes)
    for devid in devs:
        for coid in range(colocate):
            sub_node = sub_nodes[devid * colocate + coid]
            graph.assign(sub_node, devid)
    return sub_nodes


def PASSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)
            if node.name in recompute_info and recompute_info[
                    node.name] == True:
                graph.recompute([node])
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


def PASDAP(graph: IRGraph, resource):
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
                    assert False, pred_name
            elif node.name == 'layernorm':
                if pred_name == 'TriangleAttentionNodeEnd':
                    _tp(graph, node, tp_devs, 0, 1)
                elif pred_name == 'TriangleAttentionNodeStart':
                    _tp(graph, node, tp_devs, 0, 2)
                elif pred_name == 'TriangleMultiplicationIn':
                    _tp(graph, node, tp_devs, 0, 1)
                elif pred_name == 'TriangleMultiplicationOut':
                    _tp(graph, node, tp_devs, 0, 2)
                elif pred_name == 'MSATransition':
                    _tp(graph, node, tp_devs, 0, 2)
                elif pred_name == 'OuterProductMean':
                    _tp(graph, node, tp_devs, 0, 1)
                elif pred_name == 'MSAColAttention':
                    _tp(graph, node, tp_devs, 0, 2)
                elif pred_name == 'MSARowAttentionWithPairBias':
                    _tp(graph, node, tp_devs, 0, 2)
                elif pred_name == '' or pred_name == 'PairTransition':
                    _tp(graph, node, tp_devs, 0, 1)
                else:
                    assert False, pred_name
            elif node.name in {'sum', 'mul', 'multi2ref'}:
                if node.name == 'multi2ref' and pred_name == 'PairTransition':
                    _tp(graph, node, tp_devs, 0, 1)
                elif node.name == 'multi2ref' and pred_name == 'MSATransition':
                    _tp(graph, node, tp_devs, 0, 2)
                else:
                    _replica(graph, node, tp_devs)
            else:
                pred_name = node.name
                if node.name == 'MSARowAttentionWithPairBias':
                    sub_nodes = _tp(graph, node, tp_devs, 0, 1)
                elif node.name == 'MSAColAttention':
                    sub_nodes = _tp(graph, node, tp_devs, 0, 2)
                elif node.name == 'MSATransition':
                    sub_nodes = _tp(graph, node, tp_devs, 0, 2)
                elif node.name in {
                        'OPMLeftProj', 'OPMRightProj', 'OuterProductMean'
                }:
                    sub_nodes = _tp(graph, node, tp_devs, 0, 2)
                elif node.name in {
                        'TMOLeftProj', 'TMORightProj', 'TMOGate',
                        'TriangleMultiplicationOut'
                }:
                    sub_nodes = _tp(graph, node, tp_devs, 0, 1)
                elif node.name in {
                        'TMILeftProj', 'TMIRightProj', 'TMIGate',
                        'TriangleMultiplicationIn'
                }:
                    sub_nodes = _tp(graph, node, tp_devs, 0, 2)
                elif node.name in {'TANSBias', 'TriangleAttentionNodeStart'}:
                    sub_nodes = _tp(graph, node, tp_devs, 0, 1)
                elif node.name in {'TANEBias', 'TriangleAttentionNodeEnd'}:
                    sub_nodes = _tp(graph, node, tp_devs, 0, 2)
                elif node.name == 'PairTransition':
                    sub_nodes = _tp(graph, node, tp_devs, 0, 1)
                else:
                    assert False, node.name

                if node.name in recompute_info and recompute_info[
                        node.name] == True:
                    graph.recompute(sub_nodes)
    return graph
