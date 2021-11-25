import time

from cube.graph import IRGraph
from cube.graph.operator.function import CubeComplexEmbedding, Linear, Sum
from cube.graph.operator.function import CubeComplexFeedForward
from cube.graph.operator.function import CubeComplexSelfAttention
from cube.graph.operator.function import Transpose
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph
from cube.graph.operator.operator import IRDataOperation, IRFwOperation


def transform_policy(graph: IRGraph, resource):
    """
    The transformation policy transposes linear using tensor parallel
    """
    print('> transforming graph...')
    ndevs = resource.ngpus
    dp = 1
    tp = ndevs // dp

    # dataloader

    dnodes = [node for node in graph.nodes() if isinstance(node, IRDataOperation)]
    for dnode in dnodes:
        sub_nodes = list()
        algo = dnode.algorithms('data')
        dp_nodes = graph.partition(dnode, algo, config=dict(chunk_num=dp))
        for dp_node in dp_nodes:
            tp_nodes = graph.replicate(dp_node, times=tp)
            sub_nodes += tp_nodes
        for idx, sub_node in enumerate(sub_nodes):
            sub_node.tag = idx

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]

    # preprocess before transformer
    for fnode in fnodes[:5]:
        sub_nodes = list()
        if isinstance(fnode, CubeComplexEmbedding):
            algo = fnode.algorithms('data')
            dp_nodes = graph.partition(fnode, algo, config=dict(dim=0, chunk_num=dp))
            if dp_nodes[0].inputs(1).shape[0] >= 50000:
                for dp_node in dp_nodes:
                    algo = dp_node.algorithms('shard')
                    tp_nodes = graph.partition(dp_node, algo, config=dict(chunk_num=tp))
                    sub_nodes += tp_nodes
            else:
                for dp_node in dp_nodes:
                    tp_nodes = graph.replicate(dp_node, times=tp)
                    sub_nodes += tp_nodes
        else:
            algo = fnode.algorithms('dim')
            assert algo
            dp_nodes = graph.partition(fnode, algo, config=dict(dim=0, chunk_num=dp))
            for dp_node in dp_nodes:
                tp_nodes = graph.replicate(dp_node, times=tp)
                sub_nodes += tp_nodes
        for idx, sub_node in enumerate(sub_nodes):
            sub_node.tag = idx

    # transformers
    for fnode in fnodes[5:-3]:
        sub_nodes = list()
        if isinstance(fnode, CubeComplexSelfAttention):
            algo = fnode.algorithms('data')
            dp_nodes = graph.partition(fnode, algo, config=dict(chunk_num=dp))
            for dp_node in dp_nodes:
                algo = dp_node.algorithms('head')
                tp_nodes = graph.partition(dp_node, algo, config=dict(chunk_num=tp))
                sub_nodes += tp_nodes
        elif isinstance(fnode, CubeComplexFeedForward):
            algo = fnode.algorithms('data')
            dp_nodes = graph.partition(fnode, algo, config=dict(chunk_num=dp))
            for dp_node in dp_nodes:
                algo = dp_node.algorithms('tensor')
                tp_nodes = graph.partition(dp_node, algo, config=dict(chunk_num=tp))
                sub_nodes += tp_nodes
        else:
            # note replicate should put in the last due to bugs:
            algo = fnode.algorithms('dim')
            dp_nodes = graph.partition(fnode, algo, config=dict(dim=1, chunk_num=dp))
            for dp_node in dp_nodes:
                rep_nodes = graph.replicate(dp_node, times=tp)
                sub_nodes += rep_nodes
        for idx, sub_node in enumerate(sub_nodes):
            sub_node.tag = idx

    # post-process
    for fnode in fnodes[-3:]:
        sub_nodes = list()
        if isinstance(fnode, Transpose):
            algo = fnode.algorithms('dim')
            dp_nodes = graph.partition(fnode, algo, config=dict(dim=1, chunk_num=dp))
            for dp_node in dp_nodes:
                rep_nodes = graph.replicate(dp_node, times=tp)
                sub_nodes += rep_nodes
        elif isinstance(fnode, Linear):
            algo = fnode.algorithms('data')
            dp_nodes = graph.partition(fnode, algo, config=dict(dim=0, chunk_num=dp))
            for dp_node in dp_nodes:
                algo = dp_node.algorithms('column')
                tp_nodes = graph.partition(dp_node, algo, config=dict(chunk_num=tp))
                sub_nodes += tp_nodes
        else:
            rep_nodes = graph.replicate(fnode, times=ndevs)
            sub_nodes += rep_nodes
        for idx, sub_node in enumerate(sub_nodes):
            sub_node.tag = idx

    # print(graph)
    # assert False
    return graph


def schedule_policy(sugraph: SUGraph, resource):
    """
    The schedule policy assign devices
    """
    print('> scheduling SU...')
    start_time = time.time()

    for su in sugraph.sus():
        if su.stype == SUType.Dataloader:
            devid = su.tag[0]
            sugraph.assign(su, devid)
    print('> [scheduling] assign device...')
    for su in sugraph.fsus():
        devid = su.tag[0]
        sugraph.assign(su, devid)
        sugraph.assign(su.mirror, devid)
    fsus = sugraph.fsus()
    print('> [scheduling] setting schedule order...')
    sugraph.partial_set_order(fsus, lazy=False)

    span = time.time() - start_time
    print('> Done scheduling: {:.2f} seconds'.format(span))
    return sugraph
