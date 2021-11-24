from cube.graph import IRGraph
from cube.graph.operator.function import CubeComplexFeedForward, CubeComplexSelfAttention
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph
from cube.graph.operator.operator import IRDataOperation, IRFwOperation


def transform_policy(graph: IRGraph, resource):
    """
    The transformation policy transposes linear using tensor parallel
    """
    print('> transforming graph...')
    ndevs = resource.ngpus
    dp = 2
    tp = ndevs // dp

    # dataloader

    dnodes = [node for node in graph.nodes() if isinstance(node, IRDataOperation)]
    for dnode in dnodes:
        algo = dnode.algorithms('data')
        dp_nodes = graph.partition(dnode, algo, config=dict(chunk_num=dp))
        for idx, dp_node in enumerate(dp_nodes):
            dp_node.tag = idx * tp

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]

    for fnode in fnodes:
        sub_nodes = list()
        if isinstance(fnode, CubeComplexSelfAttention):
            algo = fnode.algorithms('data')
            dp_nodes = graph.partition(fnode, algo, config=dict(chunk_num=dp))
            for dp_node in dp_nodes:
                algo = dp_node.algorithms('head')
                tp_nodes = graph.partition(dp_node, algo, config=dict(chunk_num=tp))
                sub_nodes += tp_nodes
        elif isinstance(fnode, CubeComplexFeedForward):
            algo = fnode.algorithms('data')
            dp_nodes = graph.partition(fnode, algo, config=dict(chunk_num=dp))
            for dp_node in dp_nodes:
                algo = dp_node.algorithms('tensor')
                tp_nodes = graph.partition(dp_node, algo, config=dict(chunk_num=tp))
                sub_nodes += tp_nodes
        else:
            # note replicate should put in the last due to bugs:
            algo = fnode.algorithms('dim')
            dp_nodes = graph.partition(fnode, algo, config=dict(dim=1, chunk_num=dp))
            for dp_node in dp_nodes:
                rep_nodes = graph.replicate(dp_node, times=tp)
                sub_nodes += rep_nodes
        for idx, sub_node in enumerate(sub_nodes):
            sub_node.tag = idx

    print(graph)
    # assert False
    return graph


def schedule_policy(sugraph: SUGraph, resource):
    """
    The schedule policy assign devices
    """
    for su in sugraph.sus():
        if su.stype == SUType.Dataloader:
            devid = su.tag[0]
            sugraph.assign(su, devid)
    for su in sugraph.fsus():
        devid = su.tag[0]
        sugraph.assign(su, devid)
        sugraph.assign(su.mirror, devid)
    fsus = sugraph.fsus()
    sugraph.partial_set_order(fsus, lazy=False)
    return sugraph
