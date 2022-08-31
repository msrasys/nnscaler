from typing import List, Tuple
import numpy as np

from cube.graph import IRGraph
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.cten import IRCell
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


def _group_to_transformers(fnodes) -> List[List[IRCell]]:
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
    return transformers

# ========================= parallelisms =================================

# tensor parallelism
def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int], **configs):
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, **configs)
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

# ========================= parallelisms =================================

def PASRoundRobin(graph: IRGraph, resource):
    """
    roundrobin scheduling
    """
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    
    # group to transformer layers
    transformers = _group_to_transformers(fnodes)
    
    for lid, transformer in enumerate(transformers):
        stage_id = lid % resource.ngpus
        print(f'assigning {lid} transformer to stage {stage_id}')
        for node in transformer:
            graph.assign(node, stage_id)

    for node in graph.nodes():
        if len(node.device) == 0:
            _replica(graph, node, list(range(resource.ngpus)))

    return graph


def PAS1F1B(graph: IRGraph, resource):
    """
    1F1B scheduling
    """
    num_stage = resource.ngpus
    num_microbatch = resource.ngpus * 8
    _, stage_mesh = _create_mesh(resource.ngpus, (num_stage, 1))

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    
    # group to transformer layers
    transformers = _group_to_transformers(fnodes)

    # staging
    nlayer_per_stage = (len(transformers) // resource.ngpus)
    for lid, fnodes in enumerate(transformers):
        stage_id = min(lid // nlayer_per_stage, num_stage-1)
        print(f'assigning {lid}-th transformer layter to stage {stage_id}')
        for fnode in fnodes:
            graph.assign(fnode, stage_id)
    
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            _replica(graph, node, list(range(resource.ngpus)))

    strategy = IRSchedule1F1B(graph, num_microbatch, stage_mesh)
    graph.sched = strategy
    return graph


def PASMegatron(graph: IRGraph, resource):
    """
    1F1B scheduling
    """
    dp_size = 1
    tp_size = 2
    pp_size = resource.ngpus // (dp_size * tp_size)
    num_microbatch = resource.ngpus

    # device mesh
    dp_groups, pp_groups, tp_groups = \
        _create_mesh(resource.ngpus, (dp_size, pp_size, tp_size))
    print(f'dp groups: {dp_groups}')
    print(f'pp groups: {pp_groups}')
    print(f'tp groups: {tp_groups}')

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]

    # group to transformer layers
    transformers = _group_to_transformers(fnodes)

    # staging
    nlayer_per_stage = (len(transformers) // pp_size)
    for lid, fnodes in enumerate(transformers):
        sid = min(lid // nlayer_per_stage, pp_size-1)
        print(f'assigning {lid}-th transformer layer to stage {sid}: {tp_groups[sid]}')
        for fnode in fnodes:
            if fnode.name == 'window_attn':
                _tp(graph, fnode, tp_groups[sid], idx=1, dim=0, num=tp_size)
            elif fnode.name == 'feedforward':
                _tp(graph, fnode, tp_groups[sid], idx=1, dim=0, num=tp_size)
            else:
                _replica(graph, fnode, tp_groups[sid])
    
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            _replica(graph, node, list(range(resource.ngpus)))

    strategy = IRSchedule1F1B(graph, num_microbatch, tp_groups)
    graph.sched = strategy
    return graph
