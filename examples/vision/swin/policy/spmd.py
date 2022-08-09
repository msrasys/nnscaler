from typing import List

from cube.graph import IRGraph
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation


# ========================= parallelisms =================================

# tensor parallelism
def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int],
        idx: int, dim: int):
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
    assert sub_nodes is not None
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes

# replicate
def _replica(graph: IRGraph, node: IRFwOperation, devs: List[int]):
    sub_nodes = graph.replicate(node, times=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes

# coshard
def _coshard(graph: IRGraph, node: IRFwOperation, devs: List[int], colocate: int,
             idx: int, dim: int):
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=colocate*len(devs))
    assert sub_nodes is not None
    graph.recompute(sub_nodes)
    for devid in devs:
        for coid in range(colocate):
            sub_node = sub_nodes[devid * colocate + coid]
            graph.assign(sub_node, devid)
    return sub_nodes


# ========================= parallelisms =================================



def PASSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1
    # print(graph.extra_repr())
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)
    return graph


def PASMegatronTP(graph: IRGraph, resource):
    tp_size = resource.ngpus
    tp_devs = list(range(tp_size))
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]

    # annotating code structure -- not consider multiref on embedding weight
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]
    indices = [fnodes.index(anchor) for anchor in anchors]
    for lid, idx in enumerate(indices):
        # why -1: multiref
        fnodes[idx-1].comment = f'===> start of transformer layer {lid}'

    # attention
    attns = [node for node in fnodes if node.name == 'window_attn']
    for attn in attns:
        _tp(graph, attn, tp_devs, idx=1, dim=0)
    
    # feedforward
    ffns = [node for node in fnodes if node.name == 'feedforward']
    for ffn in ffns:
        _tp(graph, ffn, tp_devs, idx=1, dim=0)

    # replicate other nodes
    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)) and len(node.device) == 0:
            _replica(graph, node, tp_devs)

    return graph
