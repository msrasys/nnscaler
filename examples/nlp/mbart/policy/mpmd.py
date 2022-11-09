from typing import List, Tuple
import numpy as np

from cube.graph import IRGraph
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.segment import IRSegment
from cube.ir.cten import IRCell
from cube.graph.schedule.sched1f1b import IRSchedule1F1B
from cube.graph.schedule.schedmix import IRScheduleMix


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


def _group_to_blocks(fnodes) -> List[List[IRCell]]:
    """
    Grouping to [
        [Encoder Embed],
        [Encoder Layer], [Encoder Layer], ...,
        [Decoder Embed],
        [Decoder Layer], [Decoder Layer], ...
    ]
    """
    blocks = []
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]
    indices = [fnodes.index(anchor) for anchor in anchors]
    # encoder embedding
    fnodes[indices[0] + 1].comment = f'==> start of encoder embedding'
    assert anchors[0].name == 'encoder embedding'
    blocks.append(fnodes[0:indices[1]])
    indices.pop(0)
    anchors.pop(0)
    # encoder layers
    lid = 0
    while anchors[0].name == 'encoder layer':
        start, end = indices[0], indices[1]
        fnodes[start + 1].comment = f'==> start of encoder layer {lid}'
        blocks.append(fnodes[start:end])
        indices.pop(0)
        anchors.pop(0)
        lid += 1
    # decoder embedding
    assert anchors[0].name == 'decoder embedding'
    blocks.append(fnodes[indices[0]:indices[1]])
    indices.pop(0)
    anchors.pop(0)
    # decoder layers
    lid = 0
    while len(indices) != 0:
        assert anchors[0].name == 'decoder layer'
        start, end = indices[0], indices[1] if len(indices) > 1 else len(fnodes)
        fnodes[start + 1].comment = f'==> start of decoder layer {lid}'
        blocks.append(fnodes[indices[0]:end])
        indices.pop(0)
        anchors.pop(0)
        lid += 1
    return blocks


# tensor parallelism
def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int], idx: int, dim: int):
    if len(devs) == 1:
        graph.assign(node, devs[0])
        sub_nodes = [node]
    else:
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
        assert sub_nodes is not None
        for devid, sub_node in zip(devs, sub_nodes):
            graph.assign(sub_node, devid)
    return sub_nodes

# replicate
def _replica(graph: IRGraph, node: IRFwOperation, devs: List[int]):
    if len(devs) == 1:
        graph.assign(node, devs[0])
        sub_nodes = [node]
    else:
        sub_nodes = graph.replicate(node, times=len(devs))
        for devid, sub_node in zip(devs, sub_nodes):
            graph.assign(sub_node, devid)
    return sub_nodes


def PASSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1
    _ = _group_to_blocks(graph.select(ntype=IRFwOperation))
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        graph.assign(node, 0)
    return graph


def PAS1F1B(graph: IRGraph, resource):

    num_stages = resource.ngpus
    num_microbatch = 4
    recompute: bool = True

    blocks = _group_to_blocks(graph.select(ntype=IRFwOperation))
    enc_emb, enc_layers = blocks[0], blocks[1:len(blocks)//2]
    dec_emb, dec_layers = blocks[len(blocks)//2], blocks[len(blocks)//2+1:]
    if recompute:
        for block in blocks:
            graph.recompute(block)

    # staging
    fstages = [[] for _ in range(num_stages)]
    nlayer_per_stage = (len(enc_layers) + len(dec_layers)) // num_stages
    for lid, fnodes in enumerate(enc_layers + dec_layers):
        if lid == 0:
            fstages[0] += enc_emb
        elif lid == len(enc_layers):
            fstages[num_stages // 2] += dec_emb
        stage_id = min(lid // nlayer_per_stage, num_stages - 1)
        fstages[stage_id] += fnodes
    graph.staging(tuple(stage[0] for stage in fstages))

    dataloader = graph.select(ntype=IRDataOperation)[0]
    _replica(graph, dataloader, [0, num_stages // 2])

    fsegments = [seg for seg in graph.select(ntype=IRSegment, flatten=False) if seg.isfw()]
    assert len(fsegments) == num_stages, f"Not match: {len(fsegments)} != {num_stages}"
    for devid, segment in enumerate(fsegments):
        graph.assign(segment, devid)

    strategy = IRSchedule1F1B(graph, num_microbatch)
    graph.predef_sched(strategy)
    
    return graph


def PASMegatronTP(graph: IRGraph, resource):

    tp_size = resource.ngpus
    recompute: bool = True
    devs = list(range(tp_size))

    blocks = _group_to_blocks(graph.select(ntype=IRFwOperation))
    if recompute:
        for block in blocks:
            graph.recompute(block)
    
    for node in graph.select(ntype=IRFwOperation):
        if node.name == 'embedding':
            _tp(graph, node, devs, idx=1, dim=0)
        elif node.name == 'self_attention' or node.name == 'feedforward':
            _tp(graph, node, devs, idx=1, dim=0)
        elif node.name == 'cross_attention':
            _tp(graph, node, devs, idx=2, dim=0)
        else:
            _replica(graph, node, devs)
    
    dataloader = graph.select(ntype=IRDataOperation)[0]
    _replica(graph, dataloader, devs)

    return graph


def PASMegatron(graph: IRGraph, resource):

    dp_size = 2
    tp_size = 2
    pp_size = resource.ngpus // (dp_size * tp_size)
    recompute: bool = True
    num_microbatch = 16

    # device mesh
    dp_groups, pp_groups, tp_groups = \
        _create_mesh(resource.ngpus, (dp_size, pp_size, tp_size))
    print(f'dp groups: {dp_groups}')
    print(f'pp groups: {pp_groups}')
    print(f'tp groups: {tp_groups}')

    def get_device(dp_idx: int, pp_idx: int, tp_idx: int, ) -> int:
        return tp_groups[dp_idx * pp_size + pp_idx][tp_idx]
    
    blocks = _group_to_blocks(graph.select(ntype=IRFwOperation))
    enc_emb, enc_layers = blocks[0], blocks[1:len(blocks)//2]
    dec_emb, dec_layers = blocks[len(blocks)//2], blocks[len(blocks)//2+1:]
    if recompute:
        for block in blocks:
            graph.recompute(block)

    # pipelien stage
    fstages = [[] for _ in range(pp_size)]
    nlayer_per_stage = (len(enc_layers) + len(dec_layers)) // pp_size
    for lid, fnodes in enumerate(enc_layers + dec_layers):
        if lid == 0:
            fstages[0] += enc_emb
        elif lid == len(enc_layers):
            fstages[pp_size // 2] += dec_emb
        stage_id = min(lid // nlayer_per_stage, pp_size - 1)
        fstages[stage_id] += fnodes
    graph.staging(tuple(stage[0] for stage in fstages))

    # partition dataloader
    dataloader = graph.select(ntype=IRDataOperation)[0]
    bs = dataloader.output(0).shape[0]
    dls = _replica(graph, dataloader, [0]*dp_size) # graph.partition(dataloader, dataloader.algorithms('data'), num=dp_size)
    for dp_idx, dl in enumerate(dls):
        devices = [get_device(dp_idx, 0, tp_idx) for tp_idx in range(tp_size)]
        _replica(graph, dl, devices)
    
    # tp-dp partition
    fstages = [stage for stage in graph.select(ntype=IRSegment, flatten=False) if stage.isfw()]
    assert len(fstages) == pp_size
    for pp_idx, fstage in enumerate(fstages):
        for node in fstage.nodes():
            if len(node.inputs()) == 0: continue # anchor
            if node.name == 'embedding':
                nodes = _tp(graph, node, [0]*tp_size, idx=1, dim=0)
            elif node.name == 'self_attention' or node.name == 'feedforward':
                nodes = _tp(graph, node, [0]*tp_size, idx=1, dim=0)
            elif node.name == 'cross_attention':
                nodes = _tp(graph, node, [0]*tp_size, idx=2, dim=0)
            else:
                nodes = _replica(graph, node, [0]*tp_size)
            # data parallel
            for tp_idx, node in enumerate(nodes):
                dp_devices = [get_device(dp_idx, pp_idx, tp_idx) for dp_idx in range(dp_size)]
                batch_dim = node.input(0).shape.index(bs)
                _tp(graph, node, dp_devices, idx=0, dim=batch_dim)
    
    strategy = IRSchedule1F1B(graph, num_microbatch)
    graph.predef_sched(strategy)
    return graph


def PASMixPipe(graph: IRGraph, resource):

    pp_size = resource.ngpus

    blocks = _group_to_blocks(graph.select(ntype=IRFwOperation))
    enc_emb, enc_layers = blocks[0], blocks[1:len(blocks)//2]
    dec_emb, dec_layers = blocks[len(blocks)//2], blocks[len(blocks)//2+1:]

    num_microbatch = 4

    # pipelien stage
    embed_sid = [0, pp_size // 2 + 1]
    fstages = [[] for _ in range(pp_size)]
    nlayer_per_stage = (len(enc_layers) + len(dec_layers)) // pp_size
    for lid, fnodes in enumerate(enc_layers + dec_layers):
        stage_id = min(lid // nlayer_per_stage, pp_size - 1)
        fstages[stage_id] += fnodes
    fstages.insert(embed_sid[0], enc_emb)
    fstages.insert(embed_sid[1], dec_emb)
    graph.staging(tuple(stage[0] for stage in fstages))

    fstages = [stage for stage in graph.select(ntype=IRSegment, flatten=False) if stage.isfw()]
    assert len(fstages) == pp_size + 2
    
    # fully shard enmbedding
    enc_emb, dec_emb = fstages[embed_sid[0]], fstages[embed_sid[1]]
    tp_device = list(range(resource.ngpus))
    for node in enc_emb.nodes() + dec_emb.nodes():
        # skip anchor nodes
        if isinstance(node, IRGraphAnchor): continue
        # shard embedding layer to all devices
        if node.name == 'embedding':
            _tp(graph, node, tp_device, idx=1, dim=0)
        else:
            _replica(graph, node, tp_device)
    
    dataloader = graph.select(ntype=IRDataOperation)[0]
    _replica(graph, dataloader, tp_device)
    
    # pipeline stage to devices
    pipe_stages = [stage for sid, stage in enumerate(fstages) if sid not in embed_sid]
    assert len(pipe_stages) == pp_size
    for sid, stage in enumerate(pipe_stages):
        print(stage)
        graph.assign(stage, sid)

    strategy = IRScheduleMix(graph, num_microbatch)
    graph.predef_sched(strategy)

    return graph
