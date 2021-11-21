from cube.graph import IRGraph
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph
from cube.graph.operator.operator import IRFwOperation, IRDataOperation


def transform_policy(graph: IRGraph, resource):

    ndevs = resource.ngpus

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    assert len(fnodes) == 4

    linear1 = fnodes[0]
    gelu = fnodes[1]
    linear2 = fnodes[2]
    loss = fnodes[3]

    all_sub_nodes = list()

    algo = linear1.algorithms('data')
    sub_nodes = graph.partition(linear1, algo, config=dict(dim=1, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = gelu.algorithms('dim')
    sub_nodes = graph.partition(gelu, algo, config=dict(dim=1, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = linear2.algorithms('data')
    sub_nodes = graph.partition(linear2, algo, config=dict(dim=1, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = loss.algorithms('dim')
    sub_nodes = graph.partition(loss, algo, config=dict(dim=1, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    # data loader
    dataloaders = [node for node in graph.nodes() if isinstance(node, IRDataOperation)]
    for data_op in dataloaders:
        algo = data_op.algorithms('data')
        sub_nodes = graph.partition(data_op, algo, config=dict(chunk_num=ndevs))
        all_sub_nodes.append(sub_nodes)

    for sub_nodes in all_sub_nodes:
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
            devid = su.tag[0]
            sugraph.assign(su, devid)
    for su in sugraph.fsus():
        devid = su.tag[0]
        sugraph.assign(su, devid)
        sugraph.assign(su.mirror, devid)
    fsus = sugraph.fsus()
    sugraph.partial_set_order(fsus, lazy=False)
    return sugraph
