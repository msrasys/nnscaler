import math
import random

from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph


def transform_policy(graph, resource):
    """
    The transformation policy transposes linear using data parallel
    """
    from cube.graph.operator.operator import IRDataOperation, IRFwOperation
    for node in graph.nodes():
        if isinstance(node, IRDataOperation) or isinstance(node, IRFwOperation):
            algo = node.algorithms('data')
            assert algo is not None
            graph.partition(node, algo, config=dict(chunk_num=resource.ngpus))
    return graph


def schedule_policy(sugraph: SUGraph, resource):
    """
    The schedule policy
    """
    fb_seqs = list()
    for fsu in sugraph.fsus():
        for fb_seq in fb_seqs:
            for ksu in fb_seq[::-1]:
                if sugraph.happen_before(ksu, fsu):
                    fb_seq.append(fsu)
                    break
            else:
                continue
            break
        else:
            fb_seqs.append([fsu])
    
    # device assignment
    for su in sugraph.sus():
        if su.stype == SUType.Dataloader:
            sugraph.assign(su, 0)
    
    print(f'> collect {len(fb_seqs)} forward-backward sequence')
    for fb_seq in fb_seqs:
        chunk_num = int(math.ceil(len(fb_seq) / resource.ngpus))
        for idx, su in enumerate(fb_seq):
            # devid = int(idx // chunk_num)
            # devid = idx % resource.ngpus
            devid = random.randint(0, resource.ngpus - 1)
            sugraph.assign(su, devid)
            sugraph.assign(su.mirror, devid)

    # set partial order
    for fb_seq in fb_seqs:
        fb_seq += [fsu.mirror for fsu in fb_seq][::-1]

    seqs = list()
    for fb_seq in fb_seqs:
        seqs += fb_seq
    sugraph.partial_set_order(seqs)
    return sugraph
