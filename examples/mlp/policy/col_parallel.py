from cube.graph import IRGraph
from cube.graph.operator.operator import IRDataOperation, IRFwOperation


# def transform_policy(graph: IRGraph, resource):
#     """
#     The transformation policy transposes linear using column parallel
#     """
#     for node in graph.nodes():
#         if isinstance(node, IRFwOperation) or isinstance(node, IRDataOperation):
#             algo = node.algorithms('column')
#             if algo:
#                 sub_nodes = graph.partition(node, algo, config=dict(chunk_num=resource.ngpus))
#             else:
#                 sub_nodes = graph.replicate(node, times=resource.ngpus)
#             for idx, sub_node in enumerate(sub_nodes):
#                 sub_node.tag = idx
#     print(graph)
#     return graph
# 
# 
# def schedule_policy(sugraph: SUGraph, resource):
#     """
#     The schedule policy assign devices
#     """
#     # print(sugraph)
#     for su in sugraph.sus():
#         if su.stype == SUType.Dataloader:
#             devid = su.tag[0]
#             sugraph.assign(su, devid)
#             # sugraph.assign(su, list(range(resource.ngpus)))
#     for su in sugraph.fsus():
#         devid = su.tag[0]
#         sugraph.assign(su, devid)
#         if su.mirror is None:
#             print(f'error su: {su}')
#             assert False
#         sugraph.assign(su.mirror, devid)
#     fsus = sugraph.fsus()
#     print('> [scheduling] setting schedule order...')
#     sugraph.partial_set_order(fsus, lazy=False)
#     return sugraph


def PAS(graph: IRGraph, resource):
    """
    Linear Column Partition
    """
    for node in graph.nodes():
        if isinstance(node, IRFwOperation) or isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            if algo:
                sub_nodes = graph.partition(
                    node, algo, config=dict(chunk_num=resource.ngpus)
                )
            else:
                sub_nodes = [node]
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    print(graph.extra_repr())
    return graph
