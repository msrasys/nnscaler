from cube.graph import IRGraph
from cube.ir.operator import IRFwOperation


def PASReplica(graph: IRGraph, resource):
    assert resource.ngpus == 1
    print(graph.extra_repr())
    for node in graph.nodes():
        graph.assign(node, 0)
    # print(graph.extra_repr())
    return graph


def PASMegatron(graph: IRGraph, resource):

    tp_size = resource.ngpus
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    
    def tensor_parallelism(node, idx: int, dim: int, num: int):
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(node, algo, config=dict(idx=idx, dim=dim, num=num))
        assert all(isinstance(n, IRFwOperation) for n in sub_nodes), f"Fail to partition node {node}"
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
        return sub_nodes

    qkvs = [node for node in fnodes if node.name == 'attn_qkv']
    for qkv in qkvs:
        tensor_parallelism(qkv, idx=1, dim=0, num=tp_size)

    scores = [node for node in fnodes if node.name == 'attn_score']
    for score in scores:
        tensor_parallelism(score, idx=0, dim=1, num=tp_size)

    softmaxs = [node for node in fnodes if node.name == 'attn_softmax']
    for softmax in softmaxs:
        tensor_parallelism(softmax, idx=0, dim=1, num=tp_size)

    contexts = [node for node in fnodes if node.name == 'attn_context']
    for context in contexts:
        tensor_parallelism(context, idx=0, dim=1, num=tp_size)

    for node in graph.nodes():
        if isinstance(node, IRFwOperation) and len(node.device) == 0:
            rnodes = graph.replicate(node, times=tp_size)
            for idx, rnode in enumerate(rnodes):
                graph.assign(rnode, idx)
    return graph
