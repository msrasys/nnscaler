from typing import List, Tuple
import numpy as np

from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.graph.schedule.sched1f1b import IRSchedule1F1B


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


def PASRoundRobin(graph: IRGraph, resource):
    """
    1F1B scheduling
    """

    def org_transformer_layer(graph: IRGraph) -> List[List[IRFwOperation]]:
        multiref_idx = [
            fidx for fidx, node in enumerate(graph.nodes()) if \
                isinstance(node, IRFwOperation) and node.name == 'multiref'
        ]
        assert len(multiref_idx) % 2 == 0, "un-recognized transormer structure"
        transformers = []
        last_fidx = [fidx for fidx, node in enumerate(graph.nodes()) if isinstance(node, IRFwOperation)][-1]
        for idx in range(0, len(multiref_idx), 2):
            graph.nodes()[multiref_idx[idx]].comment = f'===> start of transformer {idx // 2}'
            start = multiref_idx[idx] if idx != 0 else 0
            end = multiref_idx[idx+2] if idx+2 < len(multiref_idx) else last_fidx+1
            transformers.append(graph.nodes()[start:end])
        return transformers

    transformers = org_transformer_layer(graph)
    for lid, fnodes in enumerate(transformers):
        stage_id = lid % resource.ngpus
        print(f'assigning {lid}-th transformer layter to stage {stage_id}')
        for fnode in fnodes:
            graph.assign(fnode, stage_id)
    
    for node in graph.nodes():
        if len(node.device) == 0:
            graph.assign(node, 0)
    
    return graph


def PAS1F1B(graph: IRGraph, resource):
    """
    1F1B scheduling
    """
    num_stage = resource.ngpus
    num_microbatch = resource.ngpus

    _, stage_mesh = _create_mesh(resource.ngpus, (num_stage, 1))

    def org_transformer_layer(graph: IRGraph) -> List[List[IRFwOperation]]:
        multiref_idx = [
            fidx for fidx, node in enumerate(graph.nodes()) if \
                isinstance(node, IRFwOperation) and node.name == 'multiref'
        ]
        assert len(multiref_idx) % 2 == 0, "un-recognized transormer structure"
        transformers = []
        last_fidx = [fidx for fidx, node in enumerate(graph.nodes()) if isinstance(node, IRFwOperation)][-1]
        for idx in range(0, len(multiref_idx), 2):
            graph.nodes()[multiref_idx[idx]].comment = f'===> start of transformer {idx // 2}'
            start = multiref_idx[idx] if idx != 0 else 0
            end = multiref_idx[idx+2] if idx+2 < len(multiref_idx) else last_fidx+1
            transformers.append(graph.nodes()[start:end])
        return transformers

    transformers = org_transformer_layer(graph)
    for lid, fnodes in enumerate(transformers):
        stage_id = min(lid // (len(transformers) // resource.ngpus), num_stage-1)
        print(f'assigning {lid}-th transformer layter to stage {stage_id}')
        for fnode in fnodes:
            graph.assign(fnode, stage_id)
    
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            graph.assign(node, 0)

    schedule = IRSchedule1F1B(num_microbatch, stage_mesh, recompute=False)
    graph.schedule_plan = schedule
    return graph