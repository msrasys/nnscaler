from typing import List, Tuple
import numpy as np

from cube.graph import IRGraph
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.graph.schedule.sched1f1b import IRSchedule1F1B


def _create_mesh(ngpus: int, group_num: Tuple[int]) -> Tuple[Tuple[Tuple[int]]]:
    """
    Create hybrid (nested) groups given the each group number.

    The product of group_num should be same with total devices.

    e.g., 6 device to 2 x 3 mesh will results [dim][group_id] = tuple[int]:
        ( 
            ( (0,1,2), (3,4,5) ),
            ( (0,3), (2,5), (3,6) ),
        )
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


def PASRoundRobin(graph: IRGraph, resource):
    """
    roundrobin scheduling
    """
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    
    # group to transformer layers
    transformers: List[List[IRFwOperation]] = []
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]
    indices = [fnodes.index(anchor) for anchor in anchors]
    for lid, idx in enumerate(indices):
        fnodes[idx-1].comment = f'===> start of transformer layer {lid}'
        start = idx if lid != 0 else 0
        end = indices[lid+1] if lid + 1 < len(anchors) else len(fnodes)
        transformers.append(fnodes[start:end])
    for lid in range(len(transformers) - 1):
        if transformers[lid][-1].name == 'multiref':
            node = transformers[lid].pop()
            transformers[lid+1].insert(0, node)
    
    for lid, transformer in enumerate(transformers):
        stage_id = lid % resource.ngpus
        print(f'assigning {lid} transformer to stage {stage_id}')
        for node in transformer:
            graph.assign(node, stage_id)

    for node in graph.nodes():
        if len(node.device) == 0:
            graph.assign(node, 0)
    
    # print(graph.extra_repr())
    return graph


def PAS1F1B(graph: IRGraph, resource):
    """
    1F1B scheduling
    """
    num_stage = resource.ngpus
    num_microbatch = resource.ngpus
    _, stage_mesh = _create_mesh(resource.ngpus, (num_stage, 1))

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    
    # group to transformer layers
    transformers: List[List[IRFwOperation]] = []
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]
    indices = [fnodes.index(anchor) for anchor in anchors]
    for lid, idx in enumerate(indices):
        fnodes[idx-1].comment = f'===> start of transformer layer {lid}'
        start = idx if lid != 0 else 0
        end = indices[lid+1] if lid + 1 < len(anchors) else len(fnodes)
        transformers.append(fnodes[start:end])
    for lid in range(len(transformers) - 1):
        if transformers[lid][-1].name == 'multiref':
            node = transformers[lid].pop()
            transformers[lid+1].insert(0, node)

    # staging
    nlayer_per_stage = (len(transformers) // resource.ngpus)
    for lid, fnodes in enumerate(transformers):
        stage_id = min(lid // nlayer_per_stage, num_stage-1)
        print(f'assigning {lid}-th transformer layter to stage {stage_id}')
        for fnode in fnodes:
            graph.assign(fnode, stage_id)
    
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            graph.assign(node, 0)

    schedule = IRSchedule1F1B(num_microbatch, stage_mesh, recompute=False)
    graph.schedule_plan = schedule
    return graph
