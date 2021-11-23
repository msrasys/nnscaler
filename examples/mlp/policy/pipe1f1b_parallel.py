from typing import List
import math

from cube.schedule.su import SUType, ScheduleUnit
from cube.schedule.sugraph import SUGraph
from cube.graph.operator.operator import IRDataOperation, IRFwOperation


def transform_policy(graph, resource):
    """
    The transformation policy transposes linear using data parallel
    """
    micro_batch_num = resource.ngpus
    for node in graph.nodes():
        if isinstance(node, IRDataOperation) or isinstance(node, IRFwOperation):
            algo = node.algorithms('data')
            if algo is not None:
                sub_nodes = graph.partition(node, algo, config=dict(chunk_num=micro_batch_num))
            else:
                algo = node.algorithms('dim')
                sub_nodes = graph.partition(node, algo, config=dict(dim=0, chunk_num=micro_batch_num))
            for idx, sub_node in enumerate(sub_nodes):
                sub_node.tag = idx
    return graph


def schedule_policy(sugraph: SUGraph, resource):
    """
    The schedule policy
    """
    # each device is a stage
    num_micro_batch = resource.ngpus
    num_stage = resource.ngpus

    fseqs: List[List[ScheduleUnit]] = [list() for _ in range(num_micro_batch)]
    fbseqs: List[List[ScheduleUnit]] = [list() for _ in range(num_micro_batch)]

    for fsu in sugraph.fsus():
        micro_bs_id = fsu.tag[0]
        fseqs[micro_bs_id].append(fsu)

    for micro_bs_id, fseq in enumerate(fbseqs):
        bseq = [fsu.mirror for fsu in fseq][::-1]
        fbseqs[micro_bs_id] = fseq + bseq

    print(f'> collect {len(fseqs)} forward-backward sequence')

    # fstages[micro_batch_id][stage] = fstages[micro_batch_id * num_stage + stage]
    fstages: List[List[ScheduleUnit]] = [
        list() for _ in range(num_micro_batch * num_stage)
    ]

    def f(micro_batch_id: int, stage_id: int) -> List[ScheduleUnit]:
        return fstages[micro_batch_id * num_stage + stage_id]

    def b(micro_batch_id: int, stage_id: int) -> List[ScheduleUnit]:
        fstage = f(micro_batch_id, stage_id)
        bstage = [fsu.mirror for fsu in fstage][::-1]
        return bstage
    
    # assign su to stages
    for micro_bid, fseq in enumerate(fseqs):
        chunk_num = int(len(fseq) // resource.ngpus)
        for idx, fsu in enumerate(fseq):
            stage = min(int(idx // chunk_num), num_stage - 1)
            fstages[micro_bid * num_stage + stage].append(fsu)

    # stage device assignment
    for micro_bid in range(num_micro_batch):
        for stage in range(num_stage):
            for su in f(micro_bid, stage):
                sugraph.assign(su, stage)
                sugraph.assign(su.mirror, stage)

    # device assignment
    for su in sugraph.sus():
        if su.stype == SUType.Dataloader:
            sugraph.assign(su, 0)

    # 1f1b scheduling
    seqs = list()

    # warmup
    for stage in range(num_stage):
        for mid in range(stage):
            seqs += f(mid, stage)

    # steady + cooldown:
    for mid in range(num_micro_batch):
        # enqueue backward
        for stage in range(num_stage-1, -1, -1):
            seqs += b(mid, stage)
        # enqueue forward
        for stage in range(num_stage):
            f_mid = mid + 1 + num_stage - stage
            if f_mid >= num_micro_batch:
                continue
            seqs += f(f_mid, stage)

    sugraph.partial_set_order(seqs)
    # print(sugraph)
    return sugraph
