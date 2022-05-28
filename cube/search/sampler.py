"""
Micro-batch sampler for scheduling search
"""
from typing import Callable, Dict, List, Tuple
from cube.graph.graph import IRGraph
from cube.ir.operator import IRFwOperation, IRBpOperation
from cube.ir.cten import IRCell
from cube.execplan import ExectuionPlan

from multiprocessing import Pool
import numpy as np
import time, copy, math


class Estimator:
    """
    A node cost is represented as (mem_weight, mem_activation, exec_time)
    """
    @staticmethod
    def taging(graph: IRGraph):
        for node in graph.nodes():
            # tag: (mem_weight, mem_activation, span)
            if isinstance(node, IRFwOperation):
                node.cost = (0, 1, 1)
            elif isinstance(node, IRBpOperation):
                node.cost = (0, -1, 2)
            else:
                node.cost = (0, 0, 0)

    @staticmethod
    def map2mem(node: IRCell):
        if node.cost is not None:
            mem_w, mem_a, span = node.cost
        else:
            mem_w, mem_a, span = 0, 0, 0
        return mem_w + mem_a

    @staticmethod
    def map2time(node: IRCell):
        if node.cost is not None:
            mem_w, mem_a, span = node.cost
        else:
            mem_w, mem_a, span = 0, 0, 0
        return span


class Sampler:
    """
    Schedule sampler
    """
    @staticmethod
    def sample(micro_seqs: List[List[IRCell]], n_microbatch: int, n_stage: int, n_device: int,
               ssampler: Callable, tsampler: Callable, wlimits: int, alimits: int):
        assert len(micro_seqs) == n_microbatch
        for seq in micro_seqs:
            assert len(seq) // 2 == n_stage
        graph = IRGraph([], [], [], 'search')
        flatten_nodes = list()
        for seq in micro_seqs:
            flatten_nodes += seq
        graph = IRGraph(flatten_nodes, [], [], 'search')
        # graph._nodes = flatten_nodes
        for sidx, placements in enumerate(ssampler(n_microbatch, n_stage, n_device)):
            print('seraching placement:\n', placements)
            # assign to device
            for mid in range(n_microbatch):
                for devid, fnode in zip(placements[mid], micro_seqs[mid]):
                    graph.assign(fnode, devid)

            # pruning: add dependecies for micro-batches with same device assignment
            # this pruning guarantees the optimal
            # graph.reset_dependency()
            # same_microbatch = dict()
            # for mid, placement in enumerate(placements):
            #     placement = tuple(placement)
            #     if placement not in same_microbatch:
            #         same_microbatch[placement] = list()
            #     same_microbatch[placement].append(mid)
            # for placement, mids in same_microbatch.items():
            #     if len(mids) > 1:
            #         print(f'find {mids} microbatch same, add dependency')
            #         for sid in range(len(placement)):
            #             # add forward dependency
            #             graph.add_schedule([micro_seqs[mid][sid] for mid in mids])
            #             # add backward dependency
            #             graph.add_schedule([micro_seqs[mid][sid+len(placement)] for mid in mids])

            # pruning
            graph.reset_dependency()
            forders = [[] for _ in range(n_device)]
            # n_device x n_stage
            borders = [[[] for _ in range(n_stage)] for _ in range(n_device)]
            for sid in range(n_stage):
                for mid in range(min(n_microbatch, alimits)):
                    devid = placements[mid][sid]
                    forders[devid].append((mid, sid))
                    borders[devid][n_stage - 1 - sid].append(mid)
            for devid, order in enumerate(forders):
                fseq = list()
                for mid, sid in order:
                    fseq.append(micro_seqs[mid][sid])
                graph.add_schedule(fseq)
            for devid, order in enumerate(borders):
                bseq = list()
                for sid in range(n_stage):
                    bseq += [micro_seqs[mid][n_stage-1-sid] for mid in order[sid]]
                graph.add_schedule(bseq)

            # search
            for seqs in tsampler(graph.nodes()):
                print(f'searching {len(seqs)} sequences under {sidx}-th placement')
                yield seqs


class TemporalSampler:
    """
    Temporal sampler takes nodes (List[IRCell]) as input
    """

    @staticmethod
    def btemporal(nodes: List[IRCell], bs=1):
        seqs = list()
        for seq in TemporalSampler.temporal(nodes):
            seqs.append(seq)
            if len(seqs) % bs == 0:
                yield seqs
                seqs = list()
        if len(seqs) > 0:
            yield seqs

    @staticmethod
    def temporal(nodes: List[IRCell], seq = None):
        if seq is None:
            seq = list()
        if len(nodes) == 0:
            yield seq
        # initial entry
        entry_nodes = TemporalSampler.ready_emit_set(remain=nodes, seq=seq)
        if len(entry_nodes) == 0:
            return None
        for node in entry_nodes:
            seq = seq + [node]
            nid = nodes.index(node)
            sub_nodes = nodes[:nid] + nodes[nid+1:]
            for res in TemporalSampler.temporal(sub_nodes, seq):
                if res is None:
                    continue
                yield res
            seq = seq[:-1]

    @staticmethod
    def ready_emit_set(remain: List[IRCell], seq: List[IRCell]):
        """
        Get ready-to-emit node list from remain node set
        """
        ready = list()
        for node in remain:
            satisfy = True
            for pre in node.predecessors():
                if pre not in seq:
                    satisfy = False
                    break
            if satisfy:
                if len(seq) > 0 and len(seq[-1].device) != 0 and len(node.device) != 0:
                    # pruning #1: filter out equal sequences
                    if seq[-1] not in node.predecessors():
                        if node.device[0] < seq[-1].device[0]:
                            continue
                ready.append(node)
        return ready


class SpatialSampler:
    """
    Spatial sampler takes (n_microbatch, n_stage, n_device) as input
    """

    @staticmethod
    def full(n_microbatch: int, n_stage: int, n_device: int, placement = None):
        # each device pick n_microbatch * n_stage // n_device blocks
        per_device_nblocks = n_microbatch * n_stage // n_device
        # placement each stage placement
        placement = placement if placement is not None else []

        if len(placement) == n_microbatch * n_stage:
            bucket_min = [n_microbatch * n_stage] * n_device
            for nid, devid in enumerate(placement):
                bucket_min[devid] = min(bucket_min[devid], nid)
            check = [bucket_min[idx + 1] - bucket_min[idx] for idx in range(n_device - 1)]
            if min(check) < 0:
                yield None
            else:
                yield placement
        else:
            # require strict increasing array [min(bucket) for bucket in buckets]
            # bucket_min = list(range(n_microbatch * n_stage, n_microbatch * n_stage + n_device + 1))
            bucket_cnt = [0] * n_device
            for nid, devid in enumerate(placement):
                # bucket_min[devid] = min(nid, bucket_min[devid]) if bucket_min[devid] is not None else nid
                bucket_cnt[devid] += 1
            for devid in range(n_device):
                if bucket_cnt[devid] < per_device_nblocks:
                    placement = placement + [devid]
                    for seq in SpatialSampler.full(n_microbatch, n_stage, n_device, placement):
                        if seq is None:
                            continue
                        yield seq
                    placement = placement[:-1]

    @staticmethod
    def same(n_microbatch: int, n_stage: int, n_device: int, wlimits: int):
        """
        Same spatial placement for each micro-batch
        """
        placements = []
        for _ in range(n_microbatch):
            placement = [sid % n_device for sid in range(n_stage)]
            placements.append(placement)
        yield placements

    @staticmethod
    def othogonal(n_microbatch: int, n_stage: int, n_device: int,
                  wlimits: int, status = None, placements = None):
        """
        Find othogonal plans given weight_limits

        Yield:
            List[microbatch][stage] = device (int)
        """
        # each element denotes number of block assigned
        status = np.zeros((n_device, n_stage), dtype=int) if status is None else status
        placements = [] if placements is None else placements
        # repeat to reduce space
        if len(placements) == wlimits:
            for idx in range(n_microbatch - wlimits):
                placements = placements + [copy.copy(placements[idx % wlimits])]
            yield placements
        # find othogonal placements
        elif len(placements) == 0:
            # fix the first one due to symmetric device
            placements = placements + [[sid % n_device for sid in range(n_stage)]]
            for sid in range(n_stage):
                status[sid % n_device][sid] += 1
            for seqs in SpatialSampler.othogonal(n_microbatch, n_stage, n_device,
                                                 wlimits, status, placements):
                yield seqs
        else:
            for placement in SpatialSampler.microbatch_othogonal(np.copy(status)):
                placements = placements + [placement]
                for sid, devid in enumerate(placement):
                    status[devid][sid] += 1
                for seqs in SpatialSampler.othogonal(n_microbatch, n_stage, n_device,
                                                     wlimits, status, placements):
                    yield seqs
                for sid, devid in enumerate(placement):
                    status[devid][sid] -= 1
                placements = placements[:-1]

    @staticmethod
    def microbatch_othogonal(status: np.ndarray, placement = None):
        """
        status:
            2D array [n_device, n_stage], each element represents
            how many stage blocks are assigned.
        """
        n_device, n_stage = status.shape
        assert n_stage == 4
        placement = [] if placement is None else placement
        if len(placement) == n_stage:
            # print(placement)
            # input('>>>out')
            yield placement
        else:
            sid = len(placement)
            allocation = np.sum(status, axis=1)
            min_alloc = np.min(allocation)
            collision = status[:,sid]
            valid = list()
            for devid, coll in enumerate(collision):
                if coll != 0 or allocation[devid] != min_alloc:
                    continue
                valid.append(devid)
            for devid in valid:
                placement = placement + [devid]
                status[devid][sid] += 1
                for seq in SpatialSampler.microbatch_othogonal(status, placement):
                    yield seq
                status[devid][sid] -= 1
                placement = placement[:-1]


    @staticmethod
    def microbatch_placement(n_stage: int, n_device: int,
                             wlimits: int, placement = None, wstatus = None):
        """
        Find microbatch placement
        Yield:
            List[stage] = device[int]
        """
        placement = [] if placement is None else placement
        wstatus = [0] * n_device if wstatus is None else wstatus
        if len(placement) == n_stage:
            yield placement
        else:
            for devid in range(n_device):
                if wstatus[devid] == wlimits:
                    continue
                placement = placement + [devid]
                wstatus[devid] += 1
                for seq in SpatialSampler.microbatch_placement(n_stage, n_device, wlimits, placement, wstatus):
                    yield seq
                wstatus[devid] -= 1
                placement = placement[:-1]


class Searcher:

    pool = Pool(processes=32)

    @staticmethod
    def search(seqs: List[List[IRCell]], bucket: Dict, n_worker: int = 1) -> Dict[int, Tuple[int, List]]:
        pool = Pool(processes=32)
        # memory (int) -> (time, seq)
        tic = time.time()
        per_worker_seqs = int(math.ceil(len(seqs) / n_worker))
        worker_buckets = list()
        for wid in range(n_worker):
            start = wid * per_worker_seqs
            stop = (wid + 1) * per_worker_seqs
            worker_seqs = seqs[start:stop]
            worker_buckets.append(pool.apply_async(Searcher._run, (worker_seqs,)))
        worker_buckets: List[Dict] = map(lambda buck: buck.get(), worker_buckets)
        # merge results
        for worker_bucket in worker_buckets:
            for mem, (span, seq) in worker_bucket.items():
                if mem in bucket and bucket[mem][0] <= span:
                    continue
                print(f'find better plan at mem budget {mem}: span: {span}')
                bucket[mem] = (span, seq)
        toc = time.time()
        throughput = round(len(seqs) / (toc - tic), 2)
        print(f'searched {len(seqs)} sequences... throughput: {throughput} seqs/s')
        pool.close()
        pool.join()

    @staticmethod
    def _run(seqs: List[List[IRCell]]) -> Dict[int, Tuple[int, List]]:
        """
        Worker run
        """
        bucket = dict()
        graph = IRGraph([], [], [], 'search')
        for seq in seqs:
            graph._nodes = seq
            execplan = ExectuionPlan(graph)
            span, mem = execplan.analyze(map2time=Estimator.map2time, map2mem=Estimator.map2mem)
            if mem not in bucket:
                bucket[mem] = (span, copy.copy(seq))
            elif bucket[mem][0] > span:
                bucket[mem] = (span, copy.copy(seq))
        return bucket


if __name__ == '__main__':

    for idx, placement in enumerate(SpatialSampler.othogonal(n_microbatch=4, n_stage=4, n_device=4, wlimits=2)):
        print(placement)
    print(f'total {idx+1} placements')
