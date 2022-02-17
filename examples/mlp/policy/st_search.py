import copy
from typing import Callable, List
import sys
from cube.graph.graph import IRGraph, IRFwOperation
from cube.graph.operator.operator import IRBpOperation, IRDataOperation
from cube.ir.cten import IRCell
from cube.execplan import ExectuionPlan

from multiprocessing import Pool


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


def topo_sequence(nodes: List[IRCell], seq = None):
    if seq is None:
        seq = list()
    if len(nodes) == 0:
        yield seq
    # initial entry
    entry_nodes = ready_emit_set(remain=nodes, seq=seq)
    if len(entry_nodes) == 0:
        return None
    for node in entry_nodes:
        seq = seq + [node]
        nid = nodes.index(node)
        sub_nodes = nodes[:nid] + nodes[nid+1:]
        for res in topo_sequence(sub_nodes, seq):
            if res is None:
                continue
            yield res
        seq = seq[:-1]


def topo_sequence_batch(nodes: List[IRCell], bs=1):
    seqs = list()
    for idx, seq in enumerate(topo_sequence(nodes)):
        seqs.append(seq)
        if len(seqs) % bs == 0:
            print(f'dispatch {len(seqs)} seq...')
            yield seqs
            seqs = list()
    if len(seqs) > 0:
        yield seqs


def stage_division(fnodes: List[IRCell], node: IRCell, num_stages: int) -> int:
    """
    Determine stage division
    """
    num_fnodes = len(fnodes)
    idx = fnodes.index(node)
    stage = min(idx // (num_fnodes // num_stages), num_stages - 1)
    return stage


def estimator(execplan: ExectuionPlan, map2time: Callable, map2mem: Callable):
    """
    Estimate time
    """
    max_time, max_mem = execplan.analyze(map2time=map2time, map2mem=map2mem)
    return max_time, max_mem


def worker(seqs: List[List[IRCell]], bucket_mem: List[int]):
    def map2time(node: IRCell):
        if isinstance(node, IRFwOperation):
            return 1
        if isinstance(node, IRBpOperation):
            return 2
        return 0

    def map2mem(node: IRCell):
        if isinstance(node, IRFwOperation):
            return 1
        if isinstance(node, IRBpOperation):
            return -1
        return 0

    bucket_times = list([sys.maxsize for _ in range(len(bucket_mem))])
    bucket_seqs = [None] * len(bucket_mem)
    graph = IRGraph([], [], [], 'search')
    for seq in seqs:
        graph._nodes = seq
        # graph.reset_dependency() # this needs as in other process dependency will break
        execplan = ExectuionPlan(graph)
        span, mem = execplan.analyze(map2time=map2time, map2mem=map2mem)
        bucket = bucket_mem.index(mem)
        if span < bucket_times[bucket]:
            bucket_times[bucket] = span
            bucket_seqs[bucket] = copy.copy(seq)
    return bucket_times, bucket_seqs


def PAS(graph: IRGraph, resource):
    num_microbatch = 8
    num_stages = 4
    fstages = [list() for _ in range(num_microbatch * num_stages)]

    def f(micro_batch_id: int, stage_id: int):
        return fstages[micro_batch_id * num_stages + stage_id]

    def b(micro_batch_id: int, stage_id: int):
        fstage = f(micro_batch_id, stage_id)
        bstage = [fnode.mirror for fnode in fstage][::-1]
        return bstage

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]

    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            graph.assign(node, 0)

    # split to micro batches
    for node in fnodes:
        stage = stage_division(fnodes, node, num_stages=num_stages)
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(
            node, algo, config=dict(idx=0, dim=0, num=num_microbatch))
        for mid, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, stage)
            f(mid, stage).append(sub_node)

    # pruning #2: symmetric microbatches, make micro-batch id smaller happen earlier
    for sid in range(num_stages):
        fops = list()
        bops = list()
        for mid in range(num_microbatch):
            fops += f(mid, sid)
            bops += b(mid, sid)
        assert graph.add_schedule(fops)
        assert graph.add_schedule(bops)

    def map2time(node: IRCell):
        if isinstance(node, IRFwOperation):
            return 1
        if isinstance(node, IRBpOperation):
            return 2
        return 0

    def map2mem(node: IRCell):
        if isinstance(node, IRFwOperation):
            return 1
        if isinstance(node, IRBpOperation):
            return -1
        return 0

    bucket_mem = list(range(num_microbatch * len(fnodes) // num_stages + 1))
    bucket_times = list([sys.maxsize for _ in range(len(bucket_mem))])
    bucket_seqs = [None] * len(bucket_mem)

    print('start sorting...')

    nproc = 24
    worker_samples = 1000
    pool = Pool(processes=nproc)
    for idx, seqs in enumerate(topo_sequence_batch(graph.nodes(), bs=nproc * worker_samples)):
        results = list()
        for wid in range(nproc):
            start = min(nproc * worker_samples, worker_samples* wid)
            stop = min(nproc * worker_samples, start + worker_samples)
            worker_seqs = seqs[start:stop]
            results.append(pool.apply_async(worker, (worker_seqs, bucket_mem)))
        results = map(lambda res: res.get(), results)
        # merge results
        for times, res_seqs in results:
            for mem, (span_new, span_old) in enumerate(zip(times, bucket_times)):
                if span_new < span_old:
                    print(f'find better plan at mem budget {mem}: span: {span_new}')
                    bucket_times[mem] = span_new
                    bucket_seqs[mem] = res_seqs[mem]
                    _graph = IRGraph([], [], [], 'search')
                    _graph._nodes = res_seqs[mem]
                    execplan = ExectuionPlan(_graph)
                    execplan.analyze(map2time=map2time, outfile=f'plan.mem{mem}.png')
        if (idx + 1) % 1 == 0:
            print(f'progress: searched {(idx + 1) * nproc * worker_samples} K sequences')
    if len(seqs) != worker_samples:
        num = idx + len(seqs) / (worker_samples * nproc)
        print(f'done search on {int(num * nproc * worker_samples)} K sequences')
    assert False

    # _graph = IRGraph([], [], [], 'search')
    # for idx, seq in enumerate(topo_sequence(graph.nodes())):
    #     _graph._nodes = seq
    #     execplan = ExectuionPlan(_graph)
    #     span, mem = execplan.analyze(map2time=map2time, map2mem=map2mem)
    #     bucket = bucket_mem.index(mem)
    #     
    #     # execplan.draw(outfile='out.png')
    #     # print(span, mem)
    #     # input('>>> ')
    #     if (idx + 1) % 5000 == 0:
    #         print(f'progress: searched {(idx + 1) // 1000}K seqs')
    #     
    #     if span < bucket_times[bucket]:
    #         print(f'find better plan at mem budget {mem}: span: {span}')
    #         bucket_times[bucket] = span
    #         bucket_seqs[bucket] = copy.copy(seq)
    #         execplan.analyze(map2time=map2time, outfile=f'plan.mem{mem}.png')
    # print(f'done search on {idx + 1} sequences')
    # assert False
