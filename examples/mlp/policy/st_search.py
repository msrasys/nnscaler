import copy
from os import stat
import time
from typing import List, Tuple, Dict
from cube.graph.graph import IRGraph, IRFwOperation
from cube.graph.operator.operator import IRBpOperation, IRDataOperation
from cube.ir.cten import IRCell
from cube.execplan import ExectuionPlan

from multiprocessing import Pool


class Estimator:
    """
    A node tag is represented as (mem_weight, mem_activation, exec_time)
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
    def sample(graph: IRGraph, n_microbatch: int, n_stage: int, n_worker: int, n_sample_per_worker: int):
        # spatial assignment
        fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
        assert n_microbatch * n_stage == len(fnodes), f"{n_microbatch * n_stage} != {len(fnodes)}"
        for placement in Sampler.spatial(n_microbatch, n_stage, n_stage):
            assert len(placement) == len(fnodes)
            print(placement)
            for fnode, devid in zip(fnodes, placement):
                graph.assign(fnode, devid)
            for seqs in Sampler.btemporal(graph.nodes(), bs=n_worker * n_sample_per_worker):
                yield seqs

    @staticmethod
    def btemporal(nodes: List[IRCell], bs=1):
        seqs = list()
        for idx, seq in enumerate(Sampler.temporal(nodes)):
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
        entry_nodes = Sampler.ready_emit_set(remain=nodes, seq=seq)
        if len(entry_nodes) == 0:
            return None
        for node in entry_nodes:
            seq = seq + [node]
            nid = nodes.index(node)
            sub_nodes = nodes[:nid] + nodes[nid+1:]
            for res in Sampler.temporal(sub_nodes, seq):
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

    @staticmethod
    def spatial(num_microbatch: int, num_stage: int, num_device: int, placement = None):
        # each device pick num_microbatch * num_stage // num_device blocks
        per_device_nblocks = num_microbatch * num_stage // num_device
        # placement each stage placement
        placement = placement if placement is not None else []

        if len(placement) == num_microbatch * num_stage:
            bucket_min = [num_microbatch * num_stage] * num_device
            for nid, devid in enumerate(placement):
                bucket_min[devid] = min(bucket_min[devid], nid)
            check = [bucket_min[idx + 1] - bucket_min[idx] for idx in range(num_device - 1)]
            if min(check) < 0:
                yield None
            else:
                yield placement
        else:
            # require strict increasing array [min(bucket) for bucket in buckets]
            # bucket_min = list(range(num_microbatch * num_stage, num_microbatch * num_stage + num_device + 1))
            bucket_cnt = [0] * num_device
            for nid, devid in enumerate(placement):
                # bucket_min[devid] = min(nid, bucket_min[devid]) if bucket_min[devid] is not None else nid
                bucket_cnt[devid] += 1
            for devid in range(num_device):
                if bucket_cnt[devid] < per_device_nblocks:
                    placement = placement + [devid]
                    for seq in Sampler.spatial(num_microbatch, num_stage, num_device, placement):
                        if seq is None:
                            continue
                        yield seq
                    placement = placement[:-1]
                # if bucket_cnt[devid] == per_device_nblocks:
                #     continue
                # # try to place on devid
                # new_min = min(bucket_min[devid], len(placement))
                # if bucket_min[devid + 1] < new_min:
                #     continue
                # placement.append(devid)
                # print(placement)
                # input(">>>1 ")
                # if len(placement) == num_microbatch * num_stage:
                #     yield placement
                # for seq in Sampler.spatial(num_microbatch, num_stage, num_device, placement):
                #     yield seq
                # placement = placement[:-1]


class Searcher:

    @staticmethod
    def run(seqs: List[List[IRCell]]):
        # mem -> (time, seq)
        bucket = dict()
        graph = IRGraph([], [], [], 'search')
        for seq in seqs:
            graph._nodes = seq
            # graph.reset_dependency() # this needs as in other process dependency will break
            execplan = ExectuionPlan(graph)
            span, mem = execplan.analyze(map2time=Estimator.map2time, map2mem=Estimator.map2mem)
            if mem not in bucket:
                bucket[mem] = (span, copy.copy(seq))
            elif bucket[mem][0] > span:
                bucket[mem] = (span, copy.copy(seq))
        return bucket


def PAS(graph: IRGraph, resource):
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    num_microbatch = len(fnodes)
    num_stages = len(fnodes)
    print(f'num-microbatch: {num_microbatch}, num-stages: {num_stages}')

    # ============================ micro-batch / stage split ============================
    fstages = [list() for _ in range(num_microbatch * num_stages)]
    def f(micro_batch_id: int, stage_id: int):
        return fstages[micro_batch_id * num_stages + stage_id]
    def b(micro_batch_id: int, stage_id: int):
        fstage = f(micro_batch_id, stage_id)
        bstage = [fnode.mirror for fnode in fstage][::-1]
        return bstage

    def stage_division(fnodes: List[IRCell], node: IRCell, num_stages: int) -> int:
        """Determine stage division
        """
        num_fnodes = len(fnodes)
        idx = fnodes.index(node)
        stage = min(idx // (num_fnodes // num_stages), num_stages - 1)
        return stage
    # split to micro batches
    for node in fnodes:
        stage = stage_division(fnodes, node, num_stages=num_stages)
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(
            node, algo, config=dict(idx=0, dim=0, num=num_microbatch))
        for mid, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, stage)
            f(mid, stage).append(sub_node)
    # ============================ micro-batch / stage split ============================

    # pruning #2: symmetric microbatches, make micro-batch id smaller happen earlier
    # for sid in range(num_stages):
    #     fops = list()
    #     bops = list()
    #     for mid in range(num_microbatch):
    #         fops += f(mid, sid)
    #         bops += b(mid, sid)
    #     assert graph.add_schedule(fops)
    #     assert graph.add_schedule(bops)

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    graph = IRGraph([], [], [], 'search')
    graph._nodes = fnodes + [fnode.mirror for fnode in fnodes[::-1]]
    graph.reset_dependency()
    Estimator.taging(graph)

    # memory (int) -> (time, seq)
    bucket = dict()
    print('start sorting...')

    nproc, worker_samples = 32, 512
    pool = Pool(processes=nproc)
    _graph = IRGraph([], [], [], 'search')
    for idx, seqs in enumerate(Sampler.sample(graph, num_microbatch, num_stages, nproc, worker_samples)):
        tic = time.time()
        results = list()
        for wid in range(nproc):
            start = worker_samples* wid
            stop = start + worker_samples
            worker_seqs = seqs[start:stop]
            results.append(pool.apply_async(Searcher.run, (worker_seqs,)))
        results: List[Dict[int, Tuple[int, List]]] = map(lambda res: res.get(), results)
        # merge results
        for worker_bucket in results:
            for mem, (span, seq) in worker_bucket.items():
                better = False
                if mem not in bucket:
                    better = True
                elif bucket[mem][0] > span:
                    better = True
                if better:
                    print(f'find better plan at mem budget {mem}: span: {span}')
                    bucket[mem] = (span, seq)
                    _graph._nodes = seq
                    execplan = ExectuionPlan(_graph)
                    execplan.analyze(map2time=Estimator.map2time, outfile=f'plan.mem{mem}.png')
        toc = time.time()
        throughput = round(len(seqs) / (toc - tic), 2)
        if (idx + 1) % 1 == 0:
            print(f'progress: searched {(idx) * nproc * worker_samples + len(seqs)} sequences, throughput: {throughput} seqs/s')
    if len(seqs) != worker_samples:
        num = idx * nproc * worker_samples + len(seqs)
        print(f'done search on {num} sequences')
    assert False


if __name__ == '__main__':
    for idx, placement in enumerate(Sampler.spatial(3, 3, 3)):
        print(placement)
    print(f'total {idx + 1} seqs')
