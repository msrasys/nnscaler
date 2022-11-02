from typing import List, Tuple
import numpy as np

from cube.graph import IRGraph
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.cten import IRCell
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.graph.segment import IRSegment
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
        fnodes[idx+1].comment = f'===> start of transformer layer {lid}'
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

def _coshard(graph: IRGraph, node: IRFwOperation, devid: int, **configs):
    algo = node.algorithms('dim')
    if node.recompute is None:
        graph.recompute([node])
    sub_nodes = graph.partition(node, algo, **configs)
    assert sub_nodes is not None
    for sub_node in sub_nodes:
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
    Megatron policy with Data, Tensor, Pipeline Parallelism.
    """
    dp_size = 1
    tp_size = 2
    pp_size = resource.ngpus // (dp_size * tp_size)
    # note coshard will only apply to first 4 tranformer blocks
    coshard = 2
    recompute: bool = False
    num_microbatch = 8

    # device mesh
    dp_groups, pp_groups, tp_groups = \
        _create_mesh(resource.ngpus, (dp_size, pp_size, tp_size))
    print(f'dp groups: {dp_groups}')
    print(f'pp groups: {pp_groups}')
    print(f'tp groups: {tp_groups}')

    def get_device(dp_idx: int, pp_idx: int, tp_idx: int, ) -> int:
        return tp_groups[dp_idx * pp_size + pp_idx][tp_idx]

    # group to transformer layers
    transformers = _group_to_transformers(graph.select(ntype=IRFwOperation))
    if recompute:
        for transformer in transformers:
            graph.recompute(transformer)

    # group to stage: set each stage operators
    fstages = [[] for _ in range(pp_size)]
    nlayer_per_stage = (len(transformers) // pp_size)
    for lid, fnodes in enumerate(transformers):
        stage_id = min(lid // nlayer_per_stage, pp_size - 1)
        fstages[stage_id] += fnodes  
    graph.staging(tuple(stages[0] for stages in fstages))

    dataloader = graph.select(ntype=IRDataOperation)[0]
    bs = dataloader.output(0).shape[0]

    # partition dataloader
    dls = _replica(graph, dataloader, [0]*dp_size) # graph.partition(dataloader, dataloader.algorithms('data'), num=dp_size)
    for dp_idx, dl in enumerate(dls):
        # only stage 0 needs dataloader
        devices = [get_device(dp_idx, 0, tp_idx) for tp_idx in range(tp_size)]
        _replica(graph, dl, devices)

    tid = 0

    # staging
    fstages = [stage for stage in graph.select(ntype=IRSegment, flatten=False) if stage.isfw()]
    assert len(fstages) == pp_size
    nlayer_per_stage = (len(transformers) // pp_size)
    for pp_idx, fstage in enumerate(fstages):
        for fnode in fstage.nodes():
            subnodes = [fnode]
            if len(fnode.inputs()) == 0: continue # anchor
            # tensor parallel -- FIXME: current restriction needs replica happen before partition
            if fnode.name == 'window_attn' or fnode.name == 'feedforward':
                subnodes = _tp(graph, fnode, [0]*tp_size, idx=1, dim=0, num=tp_size)
            elif fnode.name == 'linear': # the last embeding linear
                subnodes = _tp(graph, fnode, [0]*tp_size, idx=1, dim=0, num=tp_size)
            else:
                subnodes = _replica(graph, fnode, [0]*tp_size)
            # data parallel
            pnodes = []
            for tp_idx, subnode in enumerate(subnodes):
                dp_devices = [get_device(dp_idx, pp_idx, tp_idx) for dp_idx in range(dp_size)]
                batch_dim = 0 if bs not in subnode.input(0).shape else subnode.input(0).shape.index(bs)
                nodes = _tp(graph, subnode, devs=dp_devices, idx=0, dim=batch_dim, num=dp_size)
                pnodes += nodes
            subnodes = pnodes
            # coshard
            if fnode.name in ['window_attn', 'feedforward']:
                if coshard > 1 and tid < 4:
                    for subnode in subnodes:
                        devid = subnode.device[0]
                        _coshard(graph, subnode, devid, idx=1, dim=0, num=coshard)
                tid = tid + 1 if fnode.name == 'window_attn' else tid

    strategy = IRSchedule1F1B(graph, num_microbatch)
    graph.predef_sched(strategy)
    return graph
