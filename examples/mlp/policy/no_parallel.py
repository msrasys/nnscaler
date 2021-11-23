from cube.graph import IRGraph
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph


def transform_policy(graph: IRGraph, resource):
    """
    The transformation policy transposes linear using column parallel
    """
    return graph


def schedule_policy(sugraph: SUGraph, resource):
    """
    The schedule policy assign devices
    """
    for su in sugraph.sus():
        if su.stype == SUType.Dataloader:
            sugraph.assign(su, 0)
    for su in sugraph.fsus():
        sugraph.assign(su, 0)
        sugraph.assign(su.mirror, 0)
    return sugraph
