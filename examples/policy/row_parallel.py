from cube.graph import IRGraph
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph
from cube.graph.operator.operator import IRFwOperation


def transform_policy(graph: IRGraph, resource):
    """
    The transformation policy transposes linear using column parallel
    """
    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            algo = node.algorithms('row')
            if algo is None:
                algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, config=dict(chunk_num=resource.ngpus))
            for idx, sub_node in enumerate(sub_nodes):
                sub_node.tag = idx
    return graph


def schedule_policy(sugraph: SUGraph, resource):
    """
    The schedule policy assign devices
    """
    for su in sugraph.sus():
        if su.stype == SUType.Dataloader:
            sugraph.assign(su, 0)
            # sugraph.assign(su, list(range(resource.ngpus)))
    for su in sugraph.fsus():
        devid = su.tag[0]
        sugraph.assign(su, devid)
        sugraph.assign(su.mirror, devid)
    return sugraph
