from torch.nn.modules import dropout
from cube.graph import IRGraph
from cube.graph.operator.function import CubeComplexFeedForward, CubeComplexSelfAttention
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph
from cube.graph.operator.operator import IRFwOperation


def transform_policy(graph: IRGraph, resource):
    """
    The transformation policy transposes linear using tensor parallel
    """
    print('> transforming graph...')
    ndevs = resource.ngpus
    dp = 2
    tp = ndevs // dp

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]

    for fnode in fnodes:
        sub_nodes = list()
        if isinstance(fnode, CubeComplexSelfAttention):
            algo = fnode.algorithms('head')
            tp_nodes = graph.partition(fnode, algo, config=dict(chunk_num=tp))
            for tp_node in tp_nodes:
                algo = tp_node.algorithms('data')
                dp_nodes = graph.partition(tp_node, algo, config=dict(chunk_num=dp))
                sub_nodes += dp_nodes
        elif isinstance(fnode, CubeComplexFeedForward):
            algo = fnode.algorithms('tensor')
            tp_nodes = graph.partition(fnode, algo, config=dict(chunk_num=tp))
            for tp_node in tp_nodes:
                algo = tp_node.algorithms('data')
                dp_nodes = graph.partition(tp_node, algo, config=dict(chunk_num=dp))
                sub_nodes += dp_nodes
        else:
            rep_nodes = graph.replicate(fnode, times=tp)
            for rep_node in rep_nodes:
                algo = rep_node.algorithms('dim')
                dp_nodes = graph.partition(rep_node, algo, config=dict(dim=1, chunk_num=dp))
                sub_nodes += dp_nodes
        for idx, sub_node in enumerate(sub_nodes):
            sub_node.tag = idx

    # print(graph)
    # assert False
    return graph


def schedule_policy(sugraph: SUGraph, resource):
    """
    The schedule policy assign devices
    """
    for su in sugraph.sus():
        if su.stype == SUType.Dataloader:
            sugraph.assign(su, 0)
    for su in sugraph.fsus():
        devid = su.tag[0]
        sugraph.assign(su, devid)
        sugraph.assign(su.mirror, devid)
    fsus = sugraph.fsus()
    sugraph.partial_set_order(fsus, lazy=False)
    return sugraph
