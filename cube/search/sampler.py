"""
Micro-batch sampler for scheduling search
"""
from typing import Callable, Dict, List, Tuple
from cube.graph.graph import IRGraph, IRFwOperation
from cube.graph.operator.operator import IRBpOperation
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
               ssampler: Callable, tsampler: Callable):
        assert len(micro_seqs) == n_microbatch
        for seq in micro_seqs:
            assert len(seq) // 2 == n_stage
        graph = IRGraph([], [], [], 'search')
        flatten_nodes = list()
        for seq in micro_seqs:
            flatten_nodes += seq
        graph._nodes = flatten_nodes
        for placements in ssampler(n_microbatch, n_stage, n_device):
            print('seraching placement:\n', placements)
            # assign to device
            for mid in range(n_microbatch):
                for devid, fnode in zip(placements[mid], micro_seqs[mid]):
                    graph.assign(fnode, devid)
            for seqs in tsampler(graph.nodes()):
                yield seqs


class TemporalSampler:
    """
    Temporal sampler takes nodes (List[IRCell]) as input
    """

    @staticmethod
    def btemporal(nodes: List[IRCell], bs=1):
        seqs = list()
        for idx, seq in enumerate(TemporalSampler.temporal(nodes)):
            seqs.append(seq)
            if len(seqs) % bs == 0:
                print(f'dispatch {len(seqs)} seq...')
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
                  wlimits: int, balance = True,  placements = None):
        """
        Find most othogonal plans given weight_limits

        Yield:
            List[microbatch][stage] = device (int)
        """
        if balance:
            nstages_per_dev = n_microbatch * n_stage // n_device
        else:
            nstages_per_dev = n_microbatch * n_stage
        # wlimits = wlimits if wlimits < n_stage else n_stage
        # placements = [] if placements is None else placements
        wstatus = [set() for _ in range(n_device)]
        bstatus = [0] * n_device
        start_slots = np.array([n_stage] * n_device, dtype=int)
        # if len(placements) == n_microbatch:
        #     yield placements
        # else:
        #     for placement in placements:
        #         for sid, devid in enumerate(placement):
        #             wstatus[devid].add(sid)
        #             start_slots[devid] = min(sid, start_slots[devid])
        #             bstatus[devid] += 1
        placements = []
        for _ in range(n_microbatch):
            placement = list()
            for sid in range(n_stage):
                # get last starting device
                for devid in np.argsort(start_slots)[::-1]:
                    if bstatus[devid] == nstages_per_dev:
                        continue
                    # try place
                    if sid not in wstatus[devid] and len(wstatus[devid]) == wlimits:
                        continue
                    placement = placement + [devid]
                    wstatus[devid].add(sid)
                    bstatus[devid] += 1
                    start_slots[devid] = min(sid, start_slots[devid])
                    break
            if len(placement) != n_stage:
                raise RuntimeError("Cannot find othogonal plans")
            placements = placements + [placement]
            # for seq in SpatialSampler.othogonal(n_microbatch, n_stage, n_device, wlimits, placements):
            #     yield seq
            # placements = placements[:-1]
        yield placements

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
                if mem in bucket and bucket[mem][0] < span:
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
