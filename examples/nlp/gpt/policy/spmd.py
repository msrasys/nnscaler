from cube.graph import IRGraph
from cube.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation


def PASSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1
    # print(graph.extra_repr())
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)
    return graph


def PASMegatronTP(graph: IRGraph, resource):
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

    # annotating code structure -- not consider multiref on embedding weight
    multirefs = [node for node in fnodes if isinstance(node, IRFwOperation) and node.name == 'multiref'][1:]
    for idx in range(0, len(multirefs), 2):
        multirefs[idx].comment = f'====> start of transformer {idx // 2}'

    # attention
    attns = [node for node in fnodes if node.name == 'self_attention']
    for attn in attns:
        tensor_parallelism(attn, idx=1, dim=0, num=tp_size)
    
    # feedforward
    ffns = [node for node in fnodes if node.name == 'feedforward']
    for ffn in ffns:
        tensor_parallelism(ffn, idx=1, dim=0, num=tp_size)

    # replicate other nodes
    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)) and len(node.device) == 0:
            rnodes = graph.replicate(node, times=tp_size)
            for idx, rnode in enumerate(rnodes):
                graph.assign(rnode, idx)

    return graph
