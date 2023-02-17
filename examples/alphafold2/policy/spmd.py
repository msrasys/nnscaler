from typing import List

from numpy import TooHardError
from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation, IRBpOperation
from cube.graph.function.anchor import IRGraphAnchor


def _replica(graph: IRGraph, node: IRFwOperation, devs: List[int]):
    sub_nodes = graph.replicate(node, times=len(devs))
    for dev_id, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, dev_id)
    return sub_nodes


def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int], idx: int,
        dim: int):
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
    assert sub_nodes is not None
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def _tps(graph: IRGraph, nodes: List[IRFwOperation], devs: List[int], idx: int,
         dim: int):
    sub_nodes = []
    for node in nodes:
        sub_nodes = sub_nodes + _tp(graph, node, devs, idx, dim)
    return sub_nodes


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


def PASSingleInference(graph: IRGraph, resource):
    assert resource.ngpus == 1

    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)

    return graph


def PASSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1

    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)

    fnodes = graph.nodes()
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]

    indices = [
        fnodes.index(anchor) for anchor in anchors
        if anchor.name == 'One Layer Evoformer Start'
        or anchor.name == 'One Layer Evoformer End'
    ]
    assert len(indices) % 2 == 0
    for i in range(len(indices) // 2):
        lhs = indices[2 * i]
        rhs = indices[2 * i + 1]

        # deepmind's default recompute strategy
        graph.recompute(fnodes[lhs + 1:rhs])

        # another strategy
        # sub_indices = []
        # for j in range(lhs + 1, rhs):
        #     if isinstance(fnodes[j], IRGraphAnchor):
        #         sub_indices.append(j)
        # sub_indices.append(rhs)
        # for j in range(len(sub_indices) - 1):
        #     graph.recompute(fnodes[sub_indices[j] + 1:sub_indices[j + 1]])

    return graph


def PASExtraSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1

    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)

    fnodes = graph.nodes()
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]

    indices = [
        fnodes.index(anchor) for anchor in anchors
        if anchor.name == 'MSACol' or anchor.name == 'One Layer Evoformer End'
    ]
    assert len(indices) % 2 == 0
    for i in range(len(indices) // 2):
        lhs = indices[2 * i]
        rhs = indices[2 * i + 1]

        graph.recompute(fnodes[lhs + 1:rhs])
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
    return graph


def PASDAP(graph: IRGraph, resource):
    tp_size = resource.ngpus
    tp_devs = list(range(tp_size))

    fnodes = graph.nodes()
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]

    indices = [
        fnodes.index(anchor) for anchor in anchors
        if anchor.name == 'One Layer Evoformer Start'
        or anchor.name == 'One Layer Evoformer End'
    ]
    assert len(indices) % 2 == 0

    for i in range(indices[0]):
        if isinstance(fnodes[i], IRDataOperation) or isinstance(
                fnodes[i], IRFwOperation):
            _replica(graph, fnodes[i], tp_devs)

    for i in range(len(indices) // 2):
        lhs, rhs = indices[2 * i], indices[2 * i + 1]
        sub_indices = []
        for j in range(lhs + 1, rhs):
            if isinstance(fnodes[j], IRGraphAnchor):
                sub_indices.append(j)
        sub_indices.append(rhs)
        # graph.recompute(fnodes[lhs:rhs])
        for j in range(len(sub_indices) - 1):
            sub_l, sub_r = sub_indices[j], sub_indices[j + 1]
            names = []
            for k in range(sub_l + 1, sub_r):
                names.append(fnodes[k].name)
            names = set(names)
            nodes = fnodes[sub_l + 1:sub_r]
            # DO NOT USE THIS
            # graph.recompute(nodes)

            if 'MSARowAttentionWithPairBias' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 1)
            elif 'MSAColAttention' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 2)
            elif 'MSATransition' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 2)
            elif 'OuterProductMean' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 2)
            elif 'TriangleMultiplicationOut' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 1)
            elif 'TriangleMultiplicationIn' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 2)
            elif 'TriangleAttentionNodeStart' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 1)
            elif 'TriangleAttentionNodeEnd' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 2)
            elif 'PairTransition' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 1)
            else:
                assert False, names

    for i in range(indices[-1] + 1, len(fnodes)):
        if isinstance(fnodes[i], IRDataOperation) or isinstance(
                fnodes[i], IRFwOperation):
            _replica(graph, fnodes[i], tp_devs)

    return graph


def PASDAPInference(graph: IRGraph, resource):
    tp_size = resource.ngpus
    tp_devs = list(range(tp_size))

    fnodes = graph.nodes()
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]

    indices = [
        fnodes.index(anchor) for anchor in anchors
        if anchor.name == 'One Layer Evoformer Start'
        or anchor.name == 'One Layer Evoformer End'
    ]
    assert len(indices) % 2 == 0

    for i in range(indices[0]):
        if isinstance(fnodes[i], IRDataOperation) or isinstance(
                fnodes[i], IRFwOperation):
            _replica(graph, fnodes[i], tp_devs)

    for i in range(len(indices) // 2):
        lhs, rhs = indices[2 * i], indices[2 * i + 1]
        sub_indices = []
        for j in range(lhs + 1, rhs):
            if isinstance(fnodes[j], IRGraphAnchor):
                sub_indices.append(j)
        sub_indices.append(rhs)
        for j in range(len(sub_indices) - 1):
            sub_l, sub_r = sub_indices[j], sub_indices[j + 1]
            names = []
            for k in range(sub_l + 1, sub_r):
                names.append(fnodes[k].name)
            names = set(names)
            nodes = fnodes[sub_l + 1:sub_r]

            if 'MSARowAttentionWithPairBias' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 1)
            elif 'MSAColAttention' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 2)
            elif 'MSATransition' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 2)
            elif 'OuterProductMean' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 2)
            elif 'TriangleMultiplicationOut' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 1)
            elif 'TriangleMultiplicationIn' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 2)
            elif 'TriangleAttentionNodeStart' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 1)
            elif 'TriangleAttentionNodeEnd' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 2)
            elif 'PairTransition' in names:
                sub_nodes = _tps(graph, nodes, tp_devs, 0, 1)
            else:
                assert False, names

    for i in range(indices[-1] + 1, len(fnodes)):
        if isinstance(fnodes[i], IRDataOperation) or isinstance(
                fnodes[i], IRFwOperation):
            _replica(graph, fnodes[i], tp_devs)

    return graph
