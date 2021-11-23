from typing import List
import math
import random

from cube.schedule.su import SUType, ScheduleUnit
from cube.schedule.sugraph import SUGraph


def transform_policy(graph, resource):
    """
    The transformation policy transposes linear using data parallel
    """
    from cube.graph.operator.operator import IRDataOperation, IRFwOperation
    for node in graph.nodes():
        if isinstance(node, IRDataOperation) or isinstance(node, IRFwOperation):
            algo = node.algorithms('data')
            if algo is not None:
                sub_nodes = graph.partition(node, algo, config=dict(chunk_num=resource.ngpus))
            else:
                algo = node.algorithms('dim')
                sub_nodes = graph.partition(node, algo, config=dict(dim=0, chunk_num=resource.ngpus))
            for idx, sub_node in enumerate(sub_nodes):
                sub_node.tag = idx
    return graph


def schedule_policy(sugraph: SUGraph, resource):
    """
    The schedule policy
    """
    fseqs: List[List[ScheduleUnit]] = [list() for _ in range(resource.ngpus)]
    fbseqs: List[List[ScheduleUnit]] = [list() for _ in range(resource.ngpus)]

    for fsu in sugraph.fsus():
        micro_bs_id = fsu.tag[0]
        fseqs[micro_bs_id].append(fsu)

    for micro_bs_id, fseq in enumerate(fbseqs):
        bseq = [fsu.mirror for fsu in fseq][::-1]
        fbseqs[micro_bs_id] = fseq + bseq
    
    # device assignment
    for su in sugraph.sus():
        if su.stype == SUType.Dataloader:
            sugraph.assign(su, 0)

    print(f'> collect {len(fseqs)} forward-backward sequence')
    for fseq in fseqs:
        chunk_num = int(math.ceil(len(fseq) / resource.ngpus))
        for idx, su in enumerate(fseq):
            # devid = int(idx // chunk_num)
            # devid = idx % resource.ngpus
            devid = random.randint(0, resource.ngpus - 1)
            sugraph.assign(su, devid)
            sugraph.assign(su.mirror, devid)

    seqs = list()
    for fb_seq in fbseqs:
        seqs += fb_seq
    sugraph.partial_set_order(seqs)
    return sugraph
