import random
from typing import Tuple
import numpy as np

from cube.graph.graph import IRGraph
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


def PAS1F1B(graph: IRGraph, resource):

    # assert resource.ngpus == 8, "should apply on 8 gpus"
    num_stage = 4
    num_tp = resource.ngpus // num_stage
    num_microbatch = resource.ngpus

    _, tp_mesh = _create_mesh(resource.ngpus, (num_stage, num_tp))
    print(f'> pipeline-tensor parallel group: {tp_mesh}')
    assert len(tp_mesh) == num_stage

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    node2stage = lambda node: min(fnodes.index(node) // (len(fnodes) // num_stage), num_stage-1)

    for idx, node in enumerate(fnodes):
        # get tensor parallel group
        sid = node2stage(node)
        tp_group = tp_mesh[sid]
        # partition
        if node.name == 'linear':
            algo = node.algorithms('dim')
            tp_nodes = graph.partition(node, algo, idx=1, dim=idx%2, num=num_tp)
        else:
            tp_nodes = graph.replicate(node, times=num_tp)
        # assign
        for devid, node in zip(tp_group, tp_nodes):
            graph.assign(node, devid)
    
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            mesh = tp_mesh[0]
            rnodes = graph.replicate(node, times=num_tp)
            for devid, rnode in zip(mesh, rnodes):
                graph.assign(rnode, devid)
    # setup schedule to 1F1B
    # schedule = IRSchedule1F1B(num_microbatch, tp_mesh, recompute=False)
    # graph.schedule_plan = schedule
    schedule = PredefinedSched.sched_1f1b(graph, num_microbatch, num_stage)
    return graph