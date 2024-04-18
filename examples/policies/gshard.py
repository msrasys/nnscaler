"""
Policy example following GShard
"""

from typing import List

from nnscaler.ir.tensor import IRSubTensor
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.function.anchor import IRGraphAnchor


def follow(graph: IRGraph, node: IRDimops, devs: List[int], idx: int, dim: int,
           nodes: List[IRDimops]) -> List[IRDimops]:
    """
    Partition nodes along one tensor dimension

    @param node IRDimops: the entry node
    @param devs List[int]: the devices
    @param idx int: entry node partition config idx
    @param dim int: entry node partition config dim
    @param nodes List[IRDimops]: partition node scopes

    @return remain_nodes List[IRDimops]: remaining nodes that are not partitioned
    """
    assert node in nodes
    algo = node.algorithms('dim')
    if not algo.satisfy(idx=idx, dim=dim, num=len(devs)): return nodes
    # tensor parallelism
    sub_nodes = graph.partition(
        node, node.algorithms('dim'), idx=idx, dim=dim, num=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    # partition successors
    nodes.remove(node)
    for oidx, tensor in enumerate(node.outputs()):
        if not isinstance(tensor, IRSubTensor): continue
        ftensor = tensor.parent
        for pdim in range(len(ftensor.shape)):
            if sub_nodes[0].output(oidx).shape[pdim] != ftensor.shape[pdim]:
                break
        else:
            continue
        for consumer, ctensor in zip(graph.consumers(ftensor), graph.ctensors(ftensor)):
            if not isinstance(consumer, IRDimops): continue
            if isinstance(consumer, IRGraphAnchor) or consumer.name == 'multiref': continue
            if consumer in nodes:
                cidx = consumer.inputs().index(ctensor)
                follow(graph, consumer, devs, cidx, pdim, nodes)
    return nodes


def PASGShard(graph: IRGraph, resource):

    for ftensor in graph.full_tensors():
        if ftensor.is_grad(): continue
        if len(graph.consumers(ftensor)) > 1:
            graph.multiref(ftensor)

    devs = list(range(resource.ngpus))

    def replicate(node):
        sub_nodes = graph.replicate(node, times=len(devs))
        for devid, sub_node in zip(devs, sub_nodes):
            graph.assign(sub_node, devid)
    
    # print(graph.extra_repr())
    
    fwops = graph.select(ntype=(IRDataOperation, IRFwOperation))
    print(f'> total fwops: {len(fwops)}')
    while len(fwops) > 0:
        fwop = fwops[0]
        if isinstance(fwop, IRGraphAnchor) or fwop.name == 'multiref':
            fwops.pop(0)
            continue
        # replicate if the node is not IRDimops
        if not isinstance(fwop, IRDimops):
            replicate(fwop)
            fwops.pop(0)
            continue
        # partition along the longest dimension
        configs = fwop.transform_space()
        configs = sorted(configs, reverse=True,
                         key=lambda config: fwop.input(config[0]).shape[config[1]])
        for (idx, dim) in configs:
            if fwop.input(idx).shape[dim] % len(devs) != 0: continue
            if fwop.algorithms('dim').satisfy(idx=idx, dim=dim, num=len(devs)):
                print(f'> policy partition: entry Fwop{fwop.cid}: {fwop.name} idx={idx}, dim={dim}')
                follow(graph, fwop, devs, idx, dim, fwops)
                print(f'> remaining fwops: {len(fwops)}')
                break
        else:
            replicate(fwop)
            fwops.pop(0)
    return graph
