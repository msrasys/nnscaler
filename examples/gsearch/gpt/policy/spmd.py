from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation


def PASReplica(graph: IRGraph, resource):
    """
    Single device test
    """
    assert resource.ngpus == 1
    print(graph.extra_repr())
    for node in graph.nodes():
        graph.assign(node, 0)
    # print(graph.extra_repr())
    return graph


def PASMegatronTP(graph: IRGraph, resource):
    """
    Megatron tensor parallelism (attention)
    """
    tp_size = resource.ngpus
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    
    def tensor_parallelism(node: IRFwOperation, comment: str = None, **configs):
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(node, algo, **configs)
        if isinstance(comment, str):
            for sub_node in sub_nodes:
                sub_node.comment = comment
        assert all(isinstance(n, IRFwOperation) for n in sub_nodes), f"Fail to partition node {node}"
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
        return sub_nodes

    # ============ Attention ===============
    qkvs = [node for node in fnodes if node.name == 'attn_qkv']
    for idx, qkv in enumerate(qkvs):
        tensor_parallelism(qkv, f'====> start of transformer {idx}', idx=1, dim=0, num=tp_size)

    scores = [node for node in fnodes if node.name == 'attn_score']
    for score in scores:
        tensor_parallelism(score, idx=0, dim=1, num=tp_size)

    softmaxs = [node for node in fnodes if node.name == 'attn_softmax']
    for softmax in softmaxs:
        tensor_parallelism(softmax, idx=0, dim=1, num=tp_size)

    dropouts = [node for node in fnodes if node.name == 'attn_dropout']
    for dropout in dropouts:
        tensor_parallelism(dropout, idx=0, dim=1, num=tp_size)

    contexts = [node for node in fnodes if node.name == 'attn_context']
    for context in contexts:
        tensor_parallelism(context, idx=0, dim=1, num=tp_size)

    dense_outs = [node for node in fnodes if node.name == 'attn_dense_out']
    for dense in dense_outs:
        tensor_parallelism(dense, idx=0, dim=2, num=tp_size)

    # ============= MLP ===================
    linear1s = [node for node in fnodes if node.name == 'mlp_linear1']
    for mlp_linear1 in linear1s:
        tensor_parallelism(mlp_linear1, idx=1, dim=0, num=tp_size)

    gelus = [node for node in fnodes if node.name == 'gelu']
    for gelu in gelus:
        tensor_parallelism(gelu, idx=0, dim=2, num=tp_size)

    linear2s = [node for node in fnodes if node.name == 'mlp_linear2']
    for mlp_linear2 in linear2s:
        tensor_parallelism(mlp_linear2, idx=0, dim=2, num=tp_size)

    # replicate others
    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)) and len(node.device) == 0:
            rnodes = graph.replicate(node, times=tp_size)
            for idx, rnode in enumerate(rnodes):
                graph.assign(rnode, idx)
    print(graph.extra_repr())
    return graph


def PASRecompute(graph: IRGraph, resource):
    """
    Recompute parallelism test
    """
    assert resource.ngpus == 1
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    graph.recompute(fnodes)
    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)):
            graph.assign(node, 0)
    return graph

