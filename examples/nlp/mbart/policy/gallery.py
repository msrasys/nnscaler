from typing import List

from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation, IRDataOperation
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.schedule.predefined import PredefinedSched
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.cten import IRCell

from examples.utils import create_mesh, tensor_parallelism, replica


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



def PASSingle(graph: IRGraph, resource, **kwargs):
    assert resource.ngpus == 1
    _ = _group_to_blocks(graph.select(ntype=IRFwOperation))
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        graph.assign(node, 0)
    return graph


def PAS1F1B(graph: IRGraph, resource, nmicros: int = 16, **kwargs):

    num_stages = resource.ngpus
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
    replica(graph, dataloader, [0, num_stages // 2])

    fsegments = [seg for seg in graph.select(ntype=IRSegment, flatten=False) if seg.isfw()]
    assert len(fsegments) == num_stages, f"Not match: {len(fsegments)} != {num_stages}"
    for devid, segment in enumerate(fsegments):
        graph.assign(segment, devid)

    strategy = PredefinedSched(graph, nmicros, num_stages)
    graph.predef_sched(strategy)
    
    return graph


def PASMegatronTP(graph: IRGraph, resource, **kwargs):
    """Megatron-way tensor parallelism"""
    devs = list(range(resource.ngpus))    
    for node in graph.select(ntype=(IRDataOperation, IRFwOperation)):
        if node.name == 'embedding':
            tensor_parallelism(graph, node, idx=1, dim=0, devs=devs)
        elif node.name == 'self_attention' or node.name == 'feedforward':
            tensor_parallelism(graph, node, idx=1, dim=0, devs=devs)
        elif node.name == 'cross_attention':
            tensor_parallelism(graph, node, idx=2, dim=0, devs=devs)
        else:
            replica(graph, node, devs)
    return graph


def PASMegatron(graph: IRGraph, resource,
                tp_size: int = 2, dp_size: int = 1,
                nmicros: int = 16, **kwargs):
    """Megatron policy for hybrid data-tensor-pipeline parallelism"""
    dp_size = 2
    tp_size = 2
    pp_size = resource.ngpus // (dp_size * tp_size)
    recompute: bool = True
    num_microbatch = nmicros

    # device mesh
    dp_groups, pp_groups, tp_groups = \
        create_mesh(resource.ngpus, (dp_size, pp_size, tp_size))
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
    replica(graph, dataloader, list(range(resource.ngpus)))
    
    # tp-dp partition
    fstages = [stage for stage in graph.select(ntype=IRSegment, flatten=False) if stage.isfw()]
    assert len(fstages) == pp_size
    for pp_idx, fstage in enumerate(fstages):
        for node in fstage.nodes():
            if len(node.inputs()) == 0: continue # anchor
            if node.name == 'embedding':
                nodes = tensor_parallelism(graph, node, idx=1, dim=0, devs=[0]*tp_size)
            elif node.name == 'self_attention' or node.name == 'feedforward':
                nodes = tensor_parallelism(graph, node, idx=1, dim=0, devs=[0]*tp_size)
            elif node.name == 'cross_attention':
                nodes = tensor_parallelism(graph, node, idx=2, dim=0, devs=[0]*tp_size)
            else:
                nodes = replica(graph, node, [0]*tp_size)
            # data parallel
            for tp_idx, node in enumerate(nodes):
                dp_devices = [get_device(dp_idx, pp_idx, tp_idx) for dp_idx in range(dp_size)]
                batch_dim = node.input(0).shape.index(bs)
                tensor_parallelism(graph, node, dp_devices, idx=0, dim=batch_dim)
    
    strategy = PredefinedSched.sched_1f1b(graph, num_microbatch)
    graph.predef_sched(strategy)
    return graph
