import random
from typing import Tuple
import numpy as np

from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.graph.schedule.predefined import PredefinedSched


def _create_mesh(ngpus: int, group_num: Tuple[int]) -> Tuple[Tuple[Tuple[int]]]:
    """
    Create hybrid (nested) groups given the each group number.

    The product of group_num should be same with total devices.
    """
    group_num = np.array(group_num)
    cnt = np.prod(group_num)
    assert cnt == ngpus, 'total device not match'
    grid = np.arange(cnt).reshape(tuple(group_num))
    dims = list(range(len(group_num)))
    outputs = []
    for dim, num in enumerate(group_num):
        remain = ngpus // num
        order = tuple(dims[:dim] + dims[dim+1:] + [dim])
        grid_dim = np.transpose(grid, order).reshape((remain,num))
        grid_dim = grid_dim.tolist()
        outputs.append(tuple(tuple(ranks) for ranks in grid_dim))
    assert len(outputs) == len(group_num)
    return tuple(outputs)


def PASRandom(graph, resource):
    """
    Random pipeline
    """
    assert len(graph.nodes()) // 2 >= resource.ngpus, "not enough operator number."
    remain_device = set(range(resource.ngpus))
    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            if len(remain_device) != 0:
                idx = random.randint(0, len(remain_device) - 1)
                device = list(remain_device)[idx]
                remain_device.remove(device)
            else:
                device = random.randint(0, resource.ngpus - 1)
            graph.assign(node, device)
        elif isinstance(node, IRDataOperation):
            device = random.randint(0, resource.ngpus - 1)
            graph.assign(node, device)
    print(graph.extra_repr())
    return graph


def PASMegatron(graph: IRGraph, resource):

    # assert resource.ngpus == 8, "should apply on 8 gpus"
    num_stage = 4
    num_tp = resource.ngpus // num_stage
    num_microbatch = resource.ngpus * 8

    _, tp_mesh = _create_mesh(resource.ngpus, (num_stage, num_tp))
    print(f'> pipeline-tensor parallel group: {tp_mesh}')
    assert len(tp_mesh) == num_stage

    linears = graph.select('linear')
    stage_start_nodes = linears[::len(linears) // num_stage]
    stage_start_nodes = stage_start_nodes[:num_stage]
    assert len(stage_start_nodes) == num_stage, f"{len(stage_start_nodes)} != {num_stage}"
    graph.staging(stage_start_nodes)

    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegs = [seg for seg in segments if seg.isfw()]
    assert len(fsegs) == num_stage

    for sid, segment in enumerate(fsegs):
        # get tensor parallel group
        tp_group = tp_mesh[sid]
        for idx, node in enumerate(segment.nodes()):
            # partition
            if node.name == 'linear':
                algo = node.algorithms('dim')
                tp_nodes = graph.partition(node, algo, idx=1, dim=idx % 2, num=num_tp)
            else:
                tp_nodes = graph.replicate(node, times=num_tp)
            # assign
            for devid, node in zip(tp_group, tp_nodes):
                graph.assign(node, devid)
    
    for dl in graph.select(ntype=IRDataOperation):
        mesh = tp_mesh[0]
        dls = graph.replicate(dl, times=num_tp)
        for devid, dl in zip(mesh, dls):
            graph.assign(dl, devid)

    # setup schedule to 1F1B
    # schedule = IRSchedule1F1B(num_microbatch, tp_mesh, recompute=False)
    # graph.schedule_plan = schedule
    if graph.train:
        schedule = PredefinedSched.sched_1f1b(graph, num_microbatch, num_stage)
    else:
        schedule = PredefinedSched.sched_infer_pipe(graph, num_microbatch, num_stage)
    return graph
