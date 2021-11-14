from cube.graph import IRGraph
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph
from cube.graph.operator.operator import IRFwOperation, IRDataOperation


def transform_policy(graph: IRGraph, resource):
    """
    The transformation policy transposes linear using data parallel
    """
    tp = 2
    dp = int(resource.ngpus // tp)
    linear_idx = 0
    for node in graph.nodes():
        # partition data loader at data dimension
        if isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, config=dict(chunk_num=dp))
            for idx, sub_node in enumerate(sub_nodes):
                sub_node.tag = idx * tp
        # partition operators first in column and then in data
        if isinstance(node, IRFwOperation):
            all_sub_nodes = list()
            if node.algorithms('column') is not None:
                if linear_idx % 2 == 0:
                    print(' ==> column partition')
                    algo = node.algorithms('column')
                else:
                    print(' ==> row partition')
                    algo = node.algorithms('row')
                sub_nodes = graph.partition(node, algo, config=dict(chunk_num=tp))
                for sub_node in sub_nodes:
                    print(' ==> data partition')
                    algo = sub_node.algorithms('data')
                    ssub_nodes = graph.partition(sub_node, algo, config=dict(chunk_num=dp))
                    all_sub_nodes += ssub_nodes
                linear_idx += 1
            else:
                algo = node.algorithms('data')
                sub_nodes = graph.partition(node, algo, config=dict(chunk_num=resource.ngpus))
                all_sub_nodes += sub_nodes
            # add tags (vdev) for node
            for idx, ssub_node in enumerate(all_sub_nodes):
                ssub_node.tag = idx
    print(graph)
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