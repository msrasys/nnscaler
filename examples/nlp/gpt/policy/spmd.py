from typing import List

from cube.graph import IRGraph
from cube.graph.function.anchor import IRGraphAnchor
from cube.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation


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

# coshard
def _coshard(graph: IRGraph, node: IRFwOperation, devs: List[int], colocate: int,
             idx: int, dim: int):
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=colocate*len(devs))
    assert sub_nodes is not None
    graph.recompute(sub_nodes)
    for devid in devs:
        for coid in range(colocate):
            sub_node = sub_nodes[devid * colocate + coid]
            graph.assign(sub_node, devid)
    return sub_nodes


# ========================= parallelisms =================================


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


def PASMegatronTP(graph: IRGraph, resource):
    tp_size = resource.ngpus
    tp_devs = list(range(tp_size))
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]

    # annotating code structure -- not consider multiref on embedding weight
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]
    indices = [fnodes.index(anchor) for anchor in anchors]
    for lid, idx in enumerate(indices):
        fnodes[idx+1].comment = f'===> start of transformer layer {lid}'

    # attention
    attns = [node for node in fnodes if node.name == 'self_attention']
    for attn in attns:
        _tp(graph, attn, tp_devs, idx=1, dim=0)
    
    # feedforward
    ffns = [node for node in fnodes if node.name == 'feedforward']
    for ffn in ffns:
        _tp(graph, ffn, tp_devs, idx=1, dim=0)

    # partition embed
    embeds = [node for node in fnodes if node.name == 'embedding']
    for embed in embeds:
        _tp(graph, embed, tp_devs, idx=1, dim=0)

    # partition last linear
    linears = [node for node in fnodes if node.name == 'linear']
    _tp(graph, linears[-1], tp_devs, idx=1, dim=0)

    # partition loss
    sums = [node for node in fnodes if node.name == 'sum']
    assert len(sums) == 1
    _tp(graph, sums[0], tp_devs, idx=0, dim=2)

    # partition add
    # adds = [node for node in fnodes if node.name == 'add']
    # for add in adds:
    #     # subnodes = _replica(graph, add, [0] * 2)
    #     # for idx, sub_node in enumerate(subnodes):
    #     #     _tp(graph, sub_node, [0,1] if idx == 0 else [2,3], idx=0, dim=1)
    #     # _tp(graph, add, tp_devs, idx=0, dim=1)
    #     subnodes = _tp(graph, add, [0] * 2, idx=0, dim=1)
    #     for idx, sub_node in enumerate(subnodes):
    #         _replica(graph, sub_node, [0,1] if idx == 0 else [2,3])
    # 
    # # partition layernorm
    # lns = [node for node in fnodes if node.name == 'layernorm']
    # assert len(lns) > 0
    # for ln in lns:
    #     # _tp(graph, ln, tp_devs, idx=0, dim=1)
    #     # subnodes = _replica(graph, ln, [0] * 2)
    #     # for idx, sub_node in enumerate(subnodes):
    #     #     _tp(graph, sub_node, [0,1] if idx == 0 else [2,3], idx=0, dim=1)
    #     subnodes = _tp(graph, ln, [0] * 2, idx=0, dim=1)
    #     for idx, sub_node in enumerate(subnodes):
    #         _replica(graph, sub_node, [0,1] if idx == 0 else [2,3])


    # replicate other nodes
    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)) and len(node.device) == 0:
            _replica(graph, node, tp_devs)

    return graph


def PASMegatronInferTP(graph: IRGraph, resource):
    tp_size = resource.ngpus
    tp_devs = list(range(tp_size))
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]

    # annotating code structure -- not consider multiref on embedding weight
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]
    indices = [fnodes.index(anchor) for anchor in anchors]
    for lid, idx in enumerate(indices):
        # why -1: multiref
        fnodes[idx - 1].comment = f'===> start of transformer layer {lid}'

    # attention
    attns = [node for node in fnodes if node.name == 'one_attention']
    for attn in attns:
        _tp(graph, attn, tp_devs, idx=3, dim=0)

    # feedforward
    ffns = [node for node in fnodes if node.name == 'feedforward']
    for ffn in ffns:
        _tp(graph, ffn, tp_devs, idx=1, dim=0)

    # first embedding linear
    first_emb_anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor) and node.name == 'first_embed']
    print(f'last_emd_anchors = {first_emb_anchors}')
    indices = [fnodes.index(anchor) for anchor in first_emb_anchors]
    for lid, idx in enumerate(indices):
        print(f'fnodes[idx+1].name = {fnodes[idx+1].name}')
        print(f'fnodes[idx+1] = {fnodes[idx + 1]}')
        first_emb_node = fnodes[idx+1]
        _tp(graph, first_emb_node, tp_devs, idx=1, dim=0)

    # last embedding linear
    last_emb_anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor) and node.name == 'last_embed']
    print(f'last_emd_anchors = {last_emb_anchors}')
    indices = [fnodes.index(anchor) for anchor in last_emb_anchors]
    for lid, idx in enumerate(indices):
        print(f'fnodes[idx+1].name = {fnodes[idx+1].name}')
        print(f'fnodes[idx+1] = {fnodes[idx + 1]}')
        last_emb_node = fnodes[idx+1]
        _tp(graph, last_emb_node, tp_devs, idx=1, dim=0)

    # replicate other nodes
    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)) and len(node.device) == 0:
            _replica(graph, node, tp_devs)

    return graph


def PASMeshShard(graph: IRGraph, resource):

    # print(graph.extra_repr())
    tp_size = resource.ngpus
    tp_devs = list(range(tp_size))
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]

    # annotating code structure -- not consider multiref on embedding weight
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]
    indices = [fnodes.index(anchor) for anchor in anchors]
    for lid, idx in enumerate(indices):
        # why -1: multiref
        fnodes[idx-1].comment = f'===> start of transformer layer {lid}'

    # attention
    attns = [node for node in fnodes if node.name == 'self_attention']
    for attn in attns:
        # _tp(graph, attn, tp_devs, idx=1, dim=0)
        _coshard(graph, attn, tp_devs, colocate=2, idx=1, dim=0)
    
    # feedforward
    ffns = [node for node in fnodes if node.name == 'feedforward']
    for ffn in ffns:
        # _tp(graph, ffn, tp_devs, idx=1, dim=0)
        _coshard(graph, ffn, tp_devs, colocate=4, idx=1, dim=0)

    # replicate other nodes
    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)) and len(node.device) == 0:
            _replica(graph, node, tp_devs)

    # print(graph.extra_repr())
    return graph

def PASMegatronWSRTP(graph: IRGraph, resource):
    tp_size = resource.ngpus
    tp_devs = list(range(tp_size))
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]

    # annotating code structure -- not consider multiref on embedding weight
    anchors = [node for node in fnodes if isinstance(node, IRGraphAnchor)]
    indices = [fnodes.index(anchor) for anchor in anchors]
    for lid, idx in enumerate(indices):
        # why -1: multiref
        fnodes[idx-1].comment = f'===> start of transformer layer {lid}'

    # attention
    
    qkvs = [node for node in fnodes if node.name == 'qkv_combined']
    #graph.recompute(qkvs)
    for qkv in qkvs:
        _tp(graph, qkv, tp_devs, idx=1, dim=0)
        

    attns = [node for node in fnodes if node.name == 'attention_mask']
    graph.recompute(attns)
    for attn in attns:
        # graph.recompute(attn)
        _tp(graph, attn, tp_devs, idx=0, dim=2)    
    # attns = [node for node in fnodes if node.name == 'self_attention']
    # for attn in attns:
    #     _tp(graph, attn, tp_devs, idx=1, dim=0)
    
    lins = [node for node in fnodes if node.name == 'lin']
    # graph.recompute(lins)
    for lin in lins:
         _tp(graph, lin, tp_devs, idx=1, dim=0)   

    # feedforward
    ffns = [node for node in fnodes if node.name == 'feedforward']
    # graph.recompute(ffns)
    for ffn in ffns:
        _tp(graph, ffn, tp_devs, idx=1, dim=0)

    # partition embed
    embeds = [node for node in fnodes if node.name == 'embedding']
    for embed in embeds:
        _tp(graph, embed, tp_devs, idx=1, dim=0)

    # partition last linear
    linears = [node for node in fnodes if node.name == 'linear']
    _tp(graph, linears[-1], tp_devs, idx=1, dim=0)

    # partition loss
    sums = [node for node in fnodes if node.name == 'sum']
    assert len(sums) == 1
    _tp(graph, sums[0], tp_devs, idx=0, dim=2)
   
    def GenerateNodesForSP(nodes):
        output=[]
        count = 0
        for node in nodes:
            if isinstance(node, (IRFwOperation)) and not isinstance(node, (IRGraphAnchor)):
                # if len(node.device) == 0:
                    sign = node.signature.split('.')[-1] 
                    cid  = node.cid
                    if len(output) == 0:
                        if sign == 'layer_norm':
                            output.append(node)
                    elif sign == 'dropout':
                        count = 0
                        output.append(node)
                        count += 1
                    elif sign == 'add' and count == 1:
                        output.append(node)
                        count += 1
                    elif sign == 'layer_norm' and count == 2:
                        output.append(node)
                    elif sign == 'add':
                        output.append(node)
        return output
    
    for node in GenerateNodesForSP(graph.nodes()):
        _tp(graph, node, tp_devs, idx=0, dim=0)
    
    # replicate other nodes
    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)) and len(node.device) == 0:
            _replica(graph, node, tp_devs)
            print(node)
            # if isinstance(node, (IRFwOperation)) and not isinstance(node, (IRGraphAnchor)):
            #     print(node.cid)

    return graph