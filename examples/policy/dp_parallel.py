from cube.graph import IRGraph
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph
from cube.graph.operator.operator import IRDataOperation, IRFwOperation


def transform_policy(graph: IRGraph, resource):
    """
    The transformation policy transposes linear using data parallel
    """
    for node in graph.nodes():            
        if isinstance(node, IRFwOperation) or isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            assert algo
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
            devid = su.tag[0]
            sugraph.assign(su, devid)
    for su in sugraph.fsus():
        devid = su.tag[0]
        sugraph.assign(su, devid)
        sugraph.assign(su.mirror, devid)
    return sugraph
