from typing import List

from nnscaler.graph import IRGraph
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.schedule.predefined import PredefinedSched
from nnscaler.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation

from examples.utils import tensor_parallelism, replica, group_to_layers

import logging
_logger = logging.getLogger(__name__)


def coshard(graph: IRGraph, node: IRFwOperation, devs: List[int], colocate: int,
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


def PASSingle(graph: IRGraph, resource, **kwargs):
    assert resource.ngpus == 1
    # print(graph.extra_repr())
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)
    return graph


def PASData(graph: IRGraph, resource, **kwargs):
    """Data parallelism"""
    devs = list(range(resource.ngpus))
    dataloader = graph.select(ntype=IRDataOperation)[0]
    bs = dataloader.output(0).shape[0]
    # replicate dataloader
    replica(graph, dataloader, devs)
    # partition forward operators
    for node in graph.select(ntype=IRFwOperation):
        if isinstance(node, IRGraphAnchor): continue
        try:
            tensor_parallelism(graph, node, idx=0, dim=0, devs=devs)
        except Exception as e:
            _logger.warning(f'fail to partition node {node.name} at idx=0, using replica')
            replica(graph, node, devs)
    return graph


def PASMegatronTP(graph: IRGraph, resource, **kwargs):
    """Megatron-way tensor parallelism"""
    devs = list(range(resource.ngpus))
    # attention
    for attn in graph.select(name='window_attn'):
        tensor_parallelism(graph, attn, idx=1, dim=0, devs=devs)
    # feedforward
    for ffn in graph.select(name='feedforward'):
        tensor_parallelism(graph, ffn, idx=1, dim=0, devs=devs)
    # replicate other nodes
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            replica(graph, node, devs)
    return graph


def PASMeshShard(graph: IRGraph, resource, **kwargs):
    """Coshard policy example"""
    devs = list(range(resource.ngpus))
    # attention
    for attn in graph.select(name='window_attn'):
        # _tp(graph, attn, tp_devs, idx=1, dim=0)
        coshard(graph, attn, devs, colocate=2, idx=1, dim=0)
    # feedforward
    for ffn in graph.select(name='feedforward'):
        # _tp(graph, ffn, tp_devs, idx=1, dim=0)
        coshard(graph, ffn, devs, colocate=4, idx=1, dim=0)
    # replicate other nodes
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            replica(graph, node, devs)
    return graph


def PAS1F1B(graph: IRGraph, resource, nmicros: int, **kwargs):
    """1F1B schedule"""
    num_stages = resource.ngpus
    num_microbatch = nmicros
    # group to transformer layers
    transformers = group_to_layers(graph.select(ntype=IRFwOperation))
    # staging
    nlayer_per_stage = (len(transformers) // resource.ngpus)
    for lid, fnodes in enumerate(transformers):
        stage_id = min(lid // nlayer_per_stage, num_stages-1)
        _logger.info(f'assigning {lid}-th transformer layter to stage {stage_id}')
        for fnode in fnodes:
            graph.assign(fnode, stage_id)
    # replicate dataloader
    for node in graph.select(ntype=IRDataOperation):
        replica(graph, node, list(range(resource.ngpus)))
    # apply 1f1b schedule
    PredefinedSched.sched_1f1b(graph, num_microbatch, num_stages)
    return graph
