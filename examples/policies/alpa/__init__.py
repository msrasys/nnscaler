from typing import List, Optional
from functools import partial
import warnings
import torch

from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.function.dimops import IRDimops, TransformRule, DimopSplit
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRFwOperation, IRDataOperation
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.graph.schedule.predefined import PredefinedSched
from nnscaler.runtime.device import DeviceGroup

from examples.policies.alpa.plan import ParallelSpec
from examples.policies.alpa.inter_op import inter_op
from examples.policies.alpa.intra_op import intra_op
from examples.policies.alpa.layer_op import annotate_structure
from examples.policies.alpa.cost_model import CostModel
from examples.policies.alpa.estimator import Estimator


def _replica(graph: IRGraph, node: IRFwOperation, devs: List[int]) -> List[IRDimops]:
    """Replicate a node"""
    sub_nodes = [node] if len(devs) == 1 else graph.replicate(node, len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def _tp(graph: IRGraph, node: IRDimops, devs: List[int], **configs) -> List[IRDimops]:
    """Tensor parallelism on a node"""
    sub_nodes = [node] if len(devs) == 1 \
        else graph.partition(node, node.algorithms('dim'), **configs)
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def _auto_multiref(graph: IRGraph, plan: ParallelSpec):
    """
    Apply automated multiref on tensors that are partitioned differently by different nodes
    """
    # get parallel strategy
    specs = dict()
    for stage in plan.stages:
        for cid, spec in stage.tp_spec.items():
            specs[cid] = spec

    for ftensor in graph.full_tensors():
        if ftensor.is_grad(): continue
        if len(graph.consumers(ftensor)) <= 1: continue
        consumers, ctensors = graph.consumers(ftensor), graph.ctensors(ftensor)
        splits = set()
        for consumer, ctensor in zip(consumers, ctensors):
            spec = specs[consumer.cid]
            if spec is None:
                splits.add(DimopSplit.R())
            else:
                idx, dim = spec
                rule: TransformRule = consumer.algorithms('dim').infer(idx, dim, 1)
                split = rule.inputs()[consumer.inputs().index(ctensor)]
                splits.add(split)
        if len(splits) > 1:
            print(f"> detected a(n) {'activation' if not ftensor.is_attr() else 'parameter'}: "
                  f"{ftensor.name}({ftensor.tid}) is partitioned differently. Apply multierf...")
            graph.multiref(ftensor)


def PASAlpa(graph: IRGraph, resource, 
            recompute: bool = False,
            nmicros: int = 1,
            db_cache: str = 'db_train.json',
            load_spec_file: Optional[str] = None,
            save_spec_file: Optional[str] = None,
            max_pp_size: Optional[int] = None,
            max_tp_size: Optional[int] = None,
            max_layer_number: int = 12) -> IRGraph:
    """
    Alpa policy examples.

    Require user to manually add cune.runtime.anchor inside model
    for AutoLayer partition position

    @param graph IRGraph: model graph
    @param resource Resource: resource
    @param recompute bool: whether to enable recompute on each layer
    @param nmicros int: number of micro-batches
    @param db_cache str: database cache file
    @param load_spec_file str: reuse spec file
    @param save_spec_file str: save spec file
    @param max_pp_size Optional[int]: limit the maximum number of pipeline parallelism size
    @param max_tp_size Optional[int]: limit the maximum number of tensor parallelism size
    @param max_layer_number Optional[int]: maximum number of layers to search
    """
    # recompute granularity will follow original anchor scope
    layers = annotate_structure(graph)
    if recompute:
        for layer in layers:
            graph.recompute(layer)
    
    anchors = graph.select(ntype=IRGraphAnchor)
    nlayers = len(anchors) + 1
    removed = 0
    while removed < nlayers - max_layer_number:
        for anchor in list(anchors[::2]):
            graph.remove(anchor)
            anchors.remove(anchor)
            removed += 1
            if removed >= nlayers - max_layer_number: break
    anchors = graph.select(ntype=IRGraphAnchor)
    if removed > 0:
        print(f'> shrink search space to {len(anchors)+1} layers')

    # enable this will follow alpa's policy: recompute on auto-layer granularity
    # layers = annotate_structure(graph)
    # if recompute:
    #     for layer in layers:
    #         graph.recompute(layer)
    nodes = tuple(graph.select(ntype=IRFwOperation))

    dl: IRDataOperation = graph.select(ntype=IRDataOperation)[0]
    mbs: int = dl.output(0).shape[dl.get_batch_dims()[0]]

    # reserve 2GB memory for nccl
    mem_limit = resource.gpus[0].memory - 2 * 1024 * 1024 * 1024
    print(f'> search [constraints]: device limitied memory: {mem_limit}')
    # profile
    print(f'> profiling model...')
    estimator = Estimator(db_cache)
    latency, memory = estimator(nodes, train=graph.train)
    print(f'> search [estimation]: single device latency: {latency} ms, memory: {memory/1024/1024/1024} GB')
    if DeviceGroup().rank == 0:
        print(f'> search [dump]: saving profiled database...')
        estimator.save()
    # build cost model
    print(f'> building cost model...')
    cost_model = CostModel(graph, estimator)
    
    # alpa search -- only apply on rank 0 to ensure deterministic
    if DeviceGroup().rank == 0:
        if isinstance(load_spec_file, str):
            print(f'loading spec from {load_spec_file}...')
            config = ParallelSpec.load(load_spec_file, graph)
        else:
            print(f'> start searching...')
            intra_solver = partial(intra_op, recompute=recompute, memory_limit=mem_limit, cost_model=cost_model)
            config = inter_op(nodes, resource.ngpus, intra_solver, mbs,
                              max_p=max_pp_size, max_t=max_tp_size)
        print(f'> parallel spec results:\n{config}')

        if isinstance(save_spec_file, str):
            print(f'> saving spec to {save_spec_file}...')
            config.save(save_spec_file)

        state: str = config.getstate()
        state = torch.tensor([ord(c) for c in state], dtype=torch.int, device=torch.cuda.current_device())
        # notify -suppose each node has 8 gpus
        for rank in range(8, DeviceGroup().world_size, 8):
            print(f'> notify rank {rank} has finished searching...')
            torch.distributed.send(torch.tensor([state.size(0)], device=torch.cuda.current_device()), dst=rank)
            torch.distributed.send(state, dst=rank)
    
    else:
        print('> waiting for rank 0 to finish searching...')
        length = torch.tensor([0], device=torch.cuda.current_device())
        torch.distributed.recv(length, src=0)
        state = torch.empty(length.item(), dtype=torch.int, device=torch.cuda.current_device())
        torch.distributed.recv(state, src=0)
        state = ''.join([chr(c) for c in state.tolist()])
        config = ParallelSpec.loadstate(state)
        print(f'> parallel spec results:\n{config}')

    print(f'> instantiate plan...')
    # print(graph.extra_repr())

    #  auto-multiref
    _auto_multiref(graph, config)

    # staging
    cid2node = {n.cid : n for n in nodes}
    leading_cids = [list(stage.tp_spec.keys())[0] for stage in config.stages]
    leading_nodes = [cid2node[cid] for cid in leading_cids]
    graph.staging(leading_nodes)
    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegments = [seg for seg in segments if seg.isfw()]
    assert len(fsegments) == len(config.stages)

    # replicate data loader
    devices = list(range(resource.ngpus))
    _replica(graph, dl, devices)

    # partition
    # TODO: make data parallel to be outside of pipeline parallelism
    for sidx, stage in enumerate(config.stages):
        tp, dp = stage.tp_size, stage.dp_size
        spec = stage.tp_spec
        stage_devices, devices = devices[:tp*dp], devices[tp*dp:]
        print(f'> applying spec: tp={tp}, dp={dp} for stage {sidx}...')
        for node in fsegments[sidx].nodes():
            if isinstance(node, IRGraphAnchor) or node.name == 'multiref':
                continue
            if node.cid not in spec:
                print(f'warning: node {node.name}({node.cid}) not in spec, replicate')
                _replica(graph, node, stage_devices)
                continue
            if mbs not in node.input(0).shape:
                if dp > 1:
                    print(f'warning: cannot find batch dimension of {node.name}({node.cid}), assuming idx=0, dim=0')
                batch_dim = 0
            else:
                batch_dim = node.input(0).shape.index(mbs)
            strategy = spec[node.cid] if node.cid in spec else None
            # data parallel
            if not isinstance(node, IRDimops):
                warnings.warn(f'detected a node {node.name} is not IRDimops, replicate for data parallel')
                dp_nodes = [node] if dp == 1 else graph.replicate(node, times=dp)
            else:
                dp_nodes = [node] if dp == 1 else \
                           graph.partition(node, node.algorithms('dim'), idx=0, dim=batch_dim, num=dp)
            # tensor parallelism
            tp_nodes = []
            for dp_node in dp_nodes:
                if strategy is None:
                    ts = [dp_node] if tp == 1 else graph.replicate(dp_node, times=tp)
                else:
                    idx, dim = strategy
                    ts = [dp_node] if tp == 1 else \
                         graph.partition(dp_node, dp_node.algorithms('dim'), idx=idx, dim=dim, num=tp)
                assert len(ts) == tp, f"got tp nodes: {ts} | partition {dp_node} with {strategy}"
                tp_nodes += ts
            for devid, tp_node in zip(stage_devices, tp_nodes):
                graph.assign(tp_node, devid)
    # print(graph.extra_repr())
    # setup schedule
    if graph.train:
        sched = PredefinedSched.sched_1f1b(graph, nmicros, len(config.stages))
    else:
        sched = PredefinedSched.sched_infer_pipe(graph, nmicros, len(config.stages))
    return graph
