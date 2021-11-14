from cube.graph import IRGraph
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph
from cube.graph.operator.operator import IRFwOperation


def transform_policy(graph: IRGraph, resource):
    """
    The transformation policy transposes linear using column parallel
    """
    linear_idx = 0
    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            algo = None
            if linear_idx % 2 == 0:
                print(f'> column partition: {node}')
                algo = node.algorithms('column')
            else:
                print(f'> row partition: {node}')
                algo = node.algorithms('row')
            if algo is None:
                print(f'> data partition: {node}')
                algo = node.algorithms('data')
            linear_idx += 1
            sub_nodes = graph.partition(node, algo, config=dict(chunk_num=resource.ngpus))
            for idx, sub_node in enumerate(sub_nodes):
                sub_node.tag = idx
    print(graph)
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
