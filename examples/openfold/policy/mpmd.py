from typing import List

from nnscaler.graph import IRGraph
from nnscaler.ir.cten import IRCell
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation
from nnscaler.graph.schedule.schednf1b import IRScheduleNF1B
from nnscaler.graph.schedule.sched1f1b import IRSchedule1F1B

import more_itertools
import numpy as np


def _group_to_evoformers(fnodes) -> List[List[IRCell]]:
    # group to evoformer layers
    evoformers: List[List[IRFwOperation]] = []
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor) and node.name == 'Evoformer Start']
    indices = [fnodes.index(anchor) for anchor in anchors]
    for lid, idx in enumerate(indices):
        # get first forward op
        for fnode in fnodes[idx+1:]:
            if not isinstance(fnode, IRGraphAnchor): break
        fnode.comment = f'===> start of evoformer layer {lid}'
        start = idx if lid != 0 else 0
        end = indices[lid+1] if lid + 1 < len(anchors) else len(fnodes)
        evoformers.append(fnodes[start:end])
    print(f'find {len(indices)} evoformer layers')
    return evoformers

# ========================= parallelisms =================================

# tensor parallelism
def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int], tag='dim', **config):
    if len(devs) == 1:
        sub_nodes = [node]
    else:
        algo = node.algorithms(tag)
        sub_nodes = graph.partition(node, algo, num=len(devs), **config)
        assert sub_nodes is not None
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, int(devid))
    return sub_nodes

# replicate
def _replica(graph: IRGraph, node: IRFwOperation, devs: List[int]):
    if len(devs) == 1:
        sub_nodes = [node]
    else:
        sub_nodes = graph.replicate(node, times=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, int(devid))
    return sub_nodes


# ========================= policies =================================


def PASSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1
    # print(graph.extra_repr())
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)
    return graph


def PASDP(graph: IRGraph, resource):
    dp_size = resource.ngpus
    dp_devs = list(range(dp_size))

    dataloader = graph.select(ntype=IRDataOperation)[0]
    bs = dataloader.output(0).shape[0]

    # partition dataloader
    dls = graph.partition(dataloader, dataloader.algorithms('data'), num=dp_size)
    for devid, dl in enumerate(dls):
        graph.assign(dl, devid)

    # partition forward operators
    for node in graph.select(ntype=IRFwOperation):
        if len(node.inputs()) == 0: continue
        #FIXME: a workaround to find batch dimension
        batch_dim = node.input(0).shape.index(bs)
        _tp(graph, node, dp_devs, idx=0, dim=batch_dim)
    
    return graph


def PASDAP(graph: IRGraph, resource, tp: int):

    assert resource.ngpus % tp == 0
    dp = resource.ngpus // tp

    devmesh = np.arange(resource.ngpus).reshape(dp, tp)
    tp_devs = list(range(tp))

    # grouping
    evoformers = _group_to_evoformers(graph.select(ntype=IRFwOperation))
    for layer in evoformers:
        graph.recompute(layer)

    dataloader: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    bs = dataloader.output(0).shape[dataloader.get_batch_dims()[0]]
    print(f'> get batch size: {bs}')
    dls: List[IRDataOperation] = _replica(graph, dataloader, tp_devs)
    for tp_idx, dl in enumerate(dls):
        dp_devs = devmesh[:,tp_idx]
        _tp(graph, dl, dp_devs, 'data')


    fnodes = graph.select(ntype=IRFwOperation)
    fnodes = [fnode for fnode in fnodes if fnode.name != 'Evoformer Start']
    
    node_groups = more_itertools.split_at(fnodes, lambda n: isinstance(n, IRGraphAnchor))
    
    for nodes in node_groups:
        # tensor parallelism
        names = set(n.name for n in nodes)
        subnodes = []
        if len(names) == 1 or 'mul' in names:  # for first layer norm operators
            for node in nodes:
                subnodes.append(_replica(graph, node, tp_devs))
        # elif 'input_packing' in names:
        #     for node in nodes:
        #         subnodes.append(_replica(graph, node, tp_devs))
        elif 'row_attn' in names:
            for node in nodes:
                subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=1))
        elif 'col_attn' in names:
            for node in nodes:
                subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=2))
        elif 'opm' in names:
            for node in nodes:
                subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=2))
        elif 'tmo' in names:
            for node in nodes:
                subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=1))
        elif 'tmi' in names:
            for node in nodes:
                subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=2))
        elif 'tri_attn_start' in names:
            for node in nodes:
                subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=1))
        elif 'tri_attn_end' in names:
            for node in nodes:
                subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=2))
        elif 'feedforward' in names:
            for node in nodes:
                subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=1))
        else:
            assert False, names
        # data parallelism
        for ns in subnodes:
            for tp_idx, subnode in enumerate(ns):
                dp_devs = devmesh[:,tp_idx]
                if bs in subnode.input(0).shape:
                    dim = subnode.input(0).shape.index(bs)
                    _tp(graph, subnode, dp_devs, idx=0, dim=dim)
                else:
                    print(f'replicate op on data parallel group: {node.name}')
                    _replica(graph, subnode, dp_devs)

    return graph


def PASRoundRobin(graph: IRGraph, resource):

    pp_size = resource.ngpus

    # grouping
    evoformers = _group_to_evoformers(graph.select(ntype=IRFwOperation))
    for layer in evoformers:
        graph.recompute(layer)

    
    fstages = [[] for _ in range(pp_size)]
    nlayer_per_stage = len(evoformers) // pp_size
    for lid, fnodes in enumerate(evoformers):
        sid = min(lid // nlayer_per_stage, pp_size - 1)
        fstages[sid] += fnodes
    graph.staging(tuple(stages[0] for stages in fstages))

    dataloader: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    graph.assign(dataloader, 0)

    fstages = [stage for stage in graph.select(ntype=IRSegment, flatten=False) if stage.isfw()]
    for sid, fstage in enumerate(fstages):
        graph.assign(fstage, sid)

    return graph


def PASNF1B(graph: IRGraph, resource, mbs: int, gbs: int, recycle: int):

    assert gbs % mbs == 0
    nmbs = gbs // mbs
    pp_size = resource.ngpus

    # grouping
    evoformers = _group_to_evoformers(graph.select(ntype=IRFwOperation))
    assert len(evoformers) % pp_size == 0
    for layer in evoformers:
        graph.recompute(layer)

    fstages = [[] for _ in range(pp_size)]
    nlayer_per_stage = len(evoformers) // pp_size
    for lid, fnodes in enumerate(evoformers):
        sid = min(lid // nlayer_per_stage, pp_size - 1)
        fstages[sid] += fnodes
    graph.staging(tuple(stages[0] for stages in fstages))

    dataloader: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    graph.assign(dataloader, 0)

    fstages = [stage for stage in graph.select(ntype=IRSegment, flatten=False) if stage.isfw()]
    for sid, fstage in enumerate(fstages):
        graph.assign(fstage, sid)

    strategy = IRSchedule1F1B(graph, nmbs)
    graph.predef_sched(strategy)

    return graph


def PASDAPPipe(graph: IRGraph, resource, mbs: int, gbs: int, tp: int, pp: int, recycle: int):

    assert gbs % mbs == 0
    assert resource.ngpus % (pp * tp) == 0
    dp = resource.ngpus // (pp * tp)
    nmbs = gbs // mbs
    
    devmesh = np.arange(resource.ngpus, dtype=int).reshape(dp, pp, tp)
    tp_devs = [0] * tp  # dummy device, which will be reset at dp


    # grouping
    evoformers = _group_to_evoformers(graph.select(ntype=IRFwOperation))
    assert len(evoformers) % pp == 0
    for layer in evoformers:
        graph.recompute(layer)

    fstages = [[] for _ in range(pp)]
    nlayer_per_stage = len(evoformers) // pp
    for lid, fnodes in enumerate(evoformers):
        sid = min(lid // nlayer_per_stage, pp - 1)
        fstages[sid] += fnodes
    graph.staging(tuple(stages[0] for stages in fstages))

    # setup dataloader
    dataloader: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    bs = dataloader.output(0).shape[dataloader.get_batch_dims()[0]]
    print(f'> get batch size: {bs}')
    dls: List[IRDataOperation] = _replica(graph, dataloader, tp_devs)
    for tp_idx, dl in enumerate(dls):
        dp_devs = devmesh[:, 0, tp_idx]
        _tp(graph, dl, dp_devs, 'data')

    fstages = [stage for stage in graph.select(ntype=IRSegment, flatten=False) if stage.isfw()]
    assert len(fstages) > 0
    for sid, fstage in enumerate(fstages):
        fnodes = fstage.select(ntype=IRFwOperation)
        fnodes = [fnode for fnode in fnodes if fnode.name != 'Evoformer Start']
        node_groups = more_itertools.split_at(fnodes, lambda n: isinstance(n, IRGraphAnchor))
        for nodes in node_groups:
            # tensor parallelism
            names = set(n.name for n in nodes)
            subnodes = []
            if len(names) == 1 or 'mul' in names:  # for first layer norm operators
                for node in nodes:
                    subnodes.append(_replica(graph, node, tp_devs))
            elif 'row_attn' in names:
                for node in nodes:
                    subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=1))
            elif 'col_attn' in names:
                for node in nodes:
                    subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=2))
            elif 'opm' in names:
                for node in nodes:
                    subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=2))
            elif 'tmo' in names:
                for node in nodes:
                    subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=1))
            elif 'tmi' in names:
                for node in nodes:
                    subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=2))
            elif 'tri_attn_start' in names:
                for node in nodes:
                    subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=1))
            elif 'tri_attn_end' in names:
                for node in nodes:
                    subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=2))
            elif 'feedforward' in names:
                for node in nodes:
                    subnodes.append(_tp(graph, node, tp_devs, idx=0, dim=1))
            else:
                assert False, names
            # data parallelism
            for ns in subnodes:
                for tp_idx, subnode in enumerate(ns):
                    dp_devs = devmesh[:, sid, tp_idx]
                    if bs in subnode.input(0).shape:
                        dim = subnode.input(0).shape.index(bs)
                        _tp(graph, subnode, dp_devs, idx=0, dim=dim)
                    else:
                        print(f'replicate op on data parallel group: {node.name}')
                        _replica(graph, subnode, dp_devs)

    strategy = IRScheduleNF1B(graph, nmbs, recycle)
    # strategy = IRSchedule1F1B(graph, nmbs)
    graph.predef_sched(strategy)

    return graph
