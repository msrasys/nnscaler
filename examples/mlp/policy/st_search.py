import copy
from typing import Callable, List
import sys
from cube.graph.graph import IRGraph, IRFwOperation
from cube.graph.operator.operator import IRBpOperation, IRDataOperation
from cube.ir.cten import IRCell
from cube.execplan import ExectuionPlan


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


def stage_division(graph: IRGraph, node: IRCell, num_stages: int) -> int:
    """
    Determine stage division
    """
    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
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

def PAS(graph: IRGraph, resource):
    num_microbatch = 4
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
        stage = stage_division(graph, node, num_stages=num_stages)
        graph.assign(node, stage)
    for node in fnodes:
        # partition at batch dimension
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(
            node, algo, config=dict(idx=0, dim=0, num=num_microbatch))
        for mid, sub_node in enumerate(sub_nodes):
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
    for idx, seq in enumerate(topo_sequence(graph.nodes())):
        # seqrepr = [node._id for node in seq]
        # print(seqrepr)
        graph._nodes = seq
        execplan = ExectuionPlan(graph)
        span, mem = execplan.analyze(map2time=map2time, map2mem=map2mem)
        bucket = bucket_mem.index(mem)
        
        # execplan.draw(outfile='out.png')
        # print(span, mem)
        # input('>>> ')
        if (idx + 1) % 5000 == 0:
            print(f'progress: searched {(idx + 1) // 1000}K seqs')
        
        if span < bucket_times[bucket]:
            print(f'find better plan at mem budget {mem}: span: {span}')
            bucket_times[bucket] = span
            bucket_seqs[bucket] = copy.copy(seq)
            execplan.analyze(map2time=map2time, outfile=f'plan.mem{mem}.png')
    print(f'done search on {idx + 1} sequences')
    assert False
