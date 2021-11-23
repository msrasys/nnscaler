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

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]

    for fnode in fnodes:
        if isinstance(fnode, CubeComplexSelfAttention):
            algo = fnode.algorithms('head')
            sub_nodes = graph.partition(fnode, algo, config=dict(chunk_num=ndevs))
        elif isinstance(fnode, CubeComplexFeedForward):
            algo = fnode.algorithms('tensor')
            sub_nodes = graph.partition(fnode, algo, config=dict(chunk_num=ndevs))
        else:
            sub_nodes = graph.replicate(fnode, ndevs)
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
