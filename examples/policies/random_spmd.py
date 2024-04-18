"""
Random SPMD policy
"""
from typing import List, Optional
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.graph.function.anchor import IRGraphAnchor
from datetime import datetime

import random


def _tp(graph: IRGraph, node: IRDimops, devs: List[int], idx: int, dim: int):
    sub_nodes = graph.partition(
        node, node.algorithms('dim'), idx=idx, dim=dim, num=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def _replica(graph: IRGraph, node, devs: List[int]):
    sub_nodes = graph.replicate(node, times=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def PASRandomSPMD(graph: IRGraph, resource, seed: Optional[int] = None):
    """
    Random SPMD policy
    """
    # get the current random state
    state = random.getstate()

    seed = int(datetime.now().timestamp()) if seed is None else seed
    print(f'> set random SPDM policy seed to {seed}')
    random.seed(seed)
    devs = list(range(resource.ngpus))

    for ftensor in graph.full_tensors():
        if ftensor.is_grad(): continue
        if len(graph.consumers(ftensor)) > 1:
            graph.multiref(ftensor)

    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
            continue
        if isinstance(node, IRDimops):
            configs = node.transform_space()
            if len(configs) == 0:
                _replica(graph, node, devs)
            else:
                configs = sorted(configs, reverse=True,
                                 key=lambda config: node.input(config[0]).shape[config[1]])
                random.shuffle(configs)
                for (idx, dim) in configs:
                    if node.input(idx).shape[dim] % len(devs) != 0: continue
                    if node.algorithms('dim').satisfy(idx=idx, dim=dim, num=len(devs)):
                        print(f'> partition node {node.name} ({node.cid}) with config idx={idx}, dim={dim}')
                        _tp(graph, node, devs, idx, dim)
                        break
                else:
                    _replica(graph, node, devs)
        else:
            _replica(graph, node, devs)

    # restore the random state
    random.setstate(state)
    return graph
