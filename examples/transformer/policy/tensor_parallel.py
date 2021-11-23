from cube.graph import IRGraph
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
    assert len(fnodes) == 23

    attn_ln = fnodes[0]

    toqkv = fnodes[1]
    q_t = fnodes[2]
    k_t = fnodes[3]
    v_t = fnodes[4]
    q_scale = fnodes[5]
    k_t2 = fnodes[6]
    qk_bmm = fnodes[7]
    mask = fnodes[8]
    softmax = fnodes[9]
    attn_dropout = fnodes[10]
    attnv_bmm = fnodes[11]
    attnview = fnodes[12]
    linear = fnodes[13]

    attn_post_dropout = fnodes[14]
    attn_residual = fnodes[15]

    ffn_ln = fnodes[16]
    ffn_linear1 = fnodes[17]
    ffn_gelu = fnodes[18]
    ffn_linear2 = fnodes[19]

    ffn_post_dropout = fnodes[20]
    ffn_post_residual = fnodes[21]

    loss = fnodes[22]


    all_sub_nodes = list()

    # ============== attention ============
    sub_nodes = graph.replicate(attn_ln, times=resource.ngpus)
    all_sub_nodes.append(sub_nodes)

    algo = toqkv.algorithms('head')
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

    algo = attn_dropout.algorithms('dim')
    sub_nodes = graph.partition(attn_dropout, algo, config=dict(dim=0, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = attnv_bmm.algorithms('data')
    sub_nodes = graph.partition(attnv_bmm, algo, config=dict(chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = attnview.algorithms('head')
    sub_nodes = graph.partition(attnview, algo, config=dict(chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = linear.algorithms('row')
    sub_nodes = graph.partition(linear, algo, config=dict(chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    # ========== between attention and mlp ===============
    sub_nodes = graph.replicate(attn_post_dropout, times=resource.ngpus)
    all_sub_nodes.append(sub_nodes)

    sub_nodes = graph.replicate(attn_residual, times=resource.ngpus)
    all_sub_nodes.append(sub_nodes)

    sub_nodes = graph.replicate(ffn_ln, times=resource.ngpus)
    all_sub_nodes.append(sub_nodes)

    # =========== mlp ===========
    algo = ffn_linear1.algorithms('column')
    sub_nodes = graph.partition(ffn_linear1, algo, config=dict(chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = ffn_gelu.algorithms('dim')
    sub_nodes = graph.partition(ffn_gelu, algo, config=dict(dim=2, chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    algo = ffn_linear2.algorithms('row')
    sub_nodes = graph.partition(ffn_linear2, algo, config=dict(chunk_num=ndevs))
    all_sub_nodes.append(sub_nodes)

    # ========== post mlp ========
    sub_nodes = graph.replicate(ffn_post_dropout, times=resource.ngpus)
    all_sub_nodes.append(sub_nodes)

    sub_nodes = graph.replicate(ffn_post_residual, times=resource.ngpus)
    all_sub_nodes.append(sub_nodes)

    # =========== loss ===========
    sub_nodes = graph.replicate(loss, times=ndevs)
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
            sugraph.assign(su, 0)
    for su in sugraph.fsus():
        devid = su.tag[0]
        sugraph.assign(su, devid)
        sugraph.assign(su.mirror, devid)
    fsus = sugraph.fsus()
    sugraph.partial_set_order(fsus, lazy=False)
    return sugraph
