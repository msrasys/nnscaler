from functools import partial
from typing import List
from cube.graph.graph import IRGraph, IRFwOperation
from cube.ir.cten import IRCell
from cube.execplan import ExectuionPlan

from cube.search.sampler import Estimator, Sampler, SpatialSampler, TemporalSampler, Searcher


class MicroBatchView:

    @staticmethod
    def node2stage(node: IRCell, fnodes: List[IRCell], n_stage: int):
        num_fnodes = len(fnodes)
        idx = fnodes.index(node)
        stage = min(idx // (num_fnodes // n_stage), n_stage - 1)
        return stage

    @staticmethod
    def split(graph: IRGraph, n_microbatch: int) -> List[IRCell]:
        """
        Split graph into micro-batch view
        """
        fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
        micro_seqs = [list() for _ in range(n_microbatch)]
        for node in fnodes:
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(
                node, algo, config=dict(idx=0, dim=0, num=n_microbatch))
            for mid, sub_node in enumerate(sub_nodes):
                micro_seqs[mid].append(sub_node)
        for mid in range(n_microbatch):
            micro_seqs[mid] = micro_seqs[mid] + [n.mirror for n in micro_seqs[mid][::-1]]
        return micro_seqs

    @staticmethod
    def flatten(micro_seqs: List[List[IRCell]]):
        flatten_nodes = list()
        for seq in micro_seqs:
            flatten_nodes += seq
        return flatten_nodes


def PAS(graph: IRGraph, resource):

    # n_microbatch, n_stage, n_device
    M, S, D = 4, 4, 4

    # memory limits
    wlimits = 2
    alimits = 4

    micro_seqs = MicroBatchView.split(graph, M)
    assert len(micro_seqs) == M and len(micro_seqs[0]) // 2 == S
    sgraph = IRGraph(MicroBatchView.flatten(micro_seqs), [], [], 'search')
    Estimator.taging(sgraph)

    n_worker, seq_per_worker = 32, 512
    tsampler = partial(TemporalSampler.btemporal, bs=n_worker*seq_per_worker)
    ssampler = partial(SpatialSampler.othogonal, wlimits=wlimits)

    bucket = dict()
    cnt = 0
    for seqs in Sampler.sample(micro_seqs, M, S, D, ssampler, tsampler):
        Searcher.search(seqs, bucket, n_worker=n_worker)
        for mem, (span, seq) in bucket.items():
            sgraph._nodes = seq
            execplan = ExectuionPlan(sgraph)
            execplan.analyze(map2time=Estimator.map2time, outfile=f'plan.mem{mem}.png')
        cnt += len(seqs)
    print(f'done search on {cnt} sequences')
    assert False


if __name__ == '__main__':
    for idx, placement in enumerate(Sampler.spatial(3, 3, 3)):
        print(placement)
    print(f'total {idx + 1} seqs')
