from typing import List

from cube.graph import IRGraph
from cube.ir.cten import IRCell
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation


def _group_to_evoformers(fnodes) -> List[List[IRCell]]:
    # group to evoformer layers
    evoformers: List[List[IRFwOperation]] = []
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]
    indices = [fnodes.index(anchor) for anchor in anchors]
    for lid, idx in enumerate(indices):
        fnodes[idx+1].comment = f'===> start of transformer layer {lid}'
        start = idx if lid != 0 else 0
        end = indices[lid+1] if lid + 1 < len(anchors) else len(fnodes)
        evoformers.append(fnodes[start:end])
    for lid in range(len(evoformers) - 1):
        if evoformers[lid][-1].name == 'multiref':
            node = evoformers[lid].pop()
            evoformers[lid+1].insert(0, node)
    return evoformers

# ========================= parallelisms =================================

# tensor parallelism
def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int],
        idx: int, dim: int, tag='dim'):
    algo = node.algorithms(tag)
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
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


def PASTP(graph: IRGraph, resource):
    tp_size = resource.ngpus
    tp_devs = list(range(tp_size))

    # grouping
    evoformers = _group_to_evoformers(graph.select(ntype=IRFwOperation))
    for layer in evoformers:
        graph.recompute(layer)

    for node in graph.select(ntype=(IRDataOperation, IRFwOperation)):
        if isinstance(node, IRGraphAnchor): continue
        if node.name == 'row_attn':
            _tp(graph, node, tp_devs, idx=2, dim=1)
        elif node.name == 'col_attn':
            _tp(graph, node, tp_devs, idx=1, dim=1)
        elif node.name == 'feedforward':
            _tp(graph, node, tp_devs, idx=1, dim=1)
        elif node.name == 'tri_attn_start':
            _tp(graph, node, tp_devs, idx=1, dim=1)
        elif node.name == 'tri_attn_end':
            _tp(graph, node, tp_devs, idx=1, dim=1)
        elif node.name == 'outer_prod_mean':
            _tp(graph, node, tp_devs, idx=0, dim=1)
        elif node.name == 'tmi_projection':
            _tp(graph, node, tp_devs, idx=0, dim=2)
        elif node.name == 'tmi_projection':
            _tp(graph, node, tp_devs, idx=0, dim=1)
        elif node.name == 'tmi_gating' or node.name == 'tmo_gating':
            _tp(graph, node, tp_devs, idx=0, dim=1)
        else:
            _replica(graph, node, tp_devs)
    return graph