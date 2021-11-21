from cube.graph import IRGraph
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph
from cube.graph.operator.operator import IRFwOperation, IRDataOperation


def transform_policy(graph: IRGraph, resource):
    """
    The transformation policy transposes linear using tensor parallel
    """
    ndevs = resource.ngpus

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    assert len(fnodes) == 14

    toqkv = fnodes[0]
    q_t = fnodes[1]
    k_t = fnodes[2]
    v_t = fnodes[3]
    q_scale = fnodes[4]
    k_t2 = fnodes[5]
    qk_bmm = fnodes[6]
    mask = fnodes[7]
    softmax = fnodes[8]
    dropout = fnodes[9]
    attnv_bmm = fnodes[10]
    attnview = fnodes[11]
    linear = fnodes[12]
    loss = fnodes[13]

    all_sub_nodes = list()

    algo = toqkv.algorithms('data')
    sub_nodes = graph.partition(toqkv, algo, config=dict(chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = q_t.algorithms('dim')
    sub_nodes = graph.partition(q_t, algo, config=dict(dim=1, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = k_t.algorithms('dim')
    sub_nodes = graph.partition(k_t, algo, config=dict(dim=1, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = v_t.algorithms('dim')
    sub_nodes = graph.partition(v_t, algo, config=dict(dim=1, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = q_scale.algorithms('dim')
    sub_nodes = graph.partition(q_scale, algo, config=dict(dim=0, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = k_t2.algorithms('dim')
    sub_nodes = graph.partition(k_t2, algo, config=dict(dim=0, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = qk_bmm.algorithms('data')
    sub_nodes = graph.partition(qk_bmm, algo, config=dict(chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = mask.algorithms('head')
    sub_nodes = graph.partition(mask, algo, config=dict(chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = softmax.algorithms('dim')
    sub_nodes = graph.partition(softmax, algo, config=dict(dim=0, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = dropout.algorithms('dim')
    sub_nodes = graph.partition(dropout, algo, config=dict(dim=0, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = attnv_bmm.algorithms('data')
    sub_nodes = graph.partition(attnv_bmm, algo, config=dict(chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = attnview.algorithms('data')
    sub_nodes = graph.partition(attnview, algo, config=dict(chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = linear.algorithms('data')
    sub_nodes = graph.partition(linear, algo, config=dict(dim=1, chunk_num=ndevs))
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
