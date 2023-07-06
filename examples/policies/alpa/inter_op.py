"""
Piper policy

https://openreview.net/attachment?id=-U9I0f2S7W&name=supplementary_material

The implementation is a little bit adapted to fit with cube's view
"""
from typing import List, Callable, Tuple, Dict, Optional
import time

from cube.ir.operator import IRFwOperation
from examples.policies.alpa.layer_op import IRLayerOp, cluster_to_layer_ops
from examples.policies.alpa.plan import StageSpec, ParallelSpec


def iter_subgraph(nodes: Tuple[IRLayerOp], s: int):
    """
    Iterate sub-graphs of the nodes

    @param nodes Tuple[IRFwOperation]
    @param s int: number of stages

    @return (sub_graph1, sub_graph2) Tuple[Tuple[IRFwOp], Tuple[IRFwOp]]
    """
    assert s > 0
    if s > 1:
        # don't consider the head and tail to be anchor
        assert len(nodes) >= s - 1, f"layer op: {len(nodes)}, stage: {s}"
        for idx in range(len(nodes)):
            remain_nodes = len(nodes) - (idx + 1)
            # sub-problem of iter(sub_graph2, s-1) must iterable
            if remain_nodes < s - 2: continue
            sub_graph1, sub_graph2 = nodes[:idx+1], nodes[idx+1:]
            yield sub_graph1, sub_graph2
    else:
        # s == 1, take all
        yield nodes, ()


def DP(nodes: Tuple[IRLayerOp], k: int, s: int, intra_solver: Callable,
       mbs: int, max_d: Optional[int] = None, max_t: Optional[int] = None,
       _cost : Dict[Tuple, float] = None,
       _config : Dict[Tuple, List[StageSpec]] = None,
       _intra_cache = None) -> Tuple[Dict, Dict]:
    """
    DP algorithm to search for balanced pipeline stage divisions by considering
    tensor parallelism and pipeline parallelism.
    
    cost[D][k][s] = min_{D' \in D} min_{t, d where t*d<=k} max( 
        TPS(D\D',t,d,s), cost[D'][k-d*t][s-1] )
    
    D: subgraph
    K: number of devices
    t: tensor parallelism size
    d: data parallelism size
    s: number of pipeline stages

    @param nodes Tuple[IRFwOperation]: sub-graph
    @param k int: number of devices
    @param s: number of pipeline stages
    @param intra_solver:
        which takes nodes, tensor parallelism size, data parallelism size
        and in-flight number of microbatches, and outputs the 
    @param mbs: micro-batch size
    @param max_d int: maximal data parallelism size constraint
    @param max_t int: maximal tensor parallelism size constraint

    @return costs Dict[( (IRCell,), k, s ), latency]
    @return config Dict[( (IRCell,), k, s ), [(IRCell,),] ]
    """
    nodes = nodes if isinstance(nodes, tuple) else tuple(nodes)
    key = (nodes, k, s)

    # initialize: dp[((), k, s)] = 0 for every k and s
    _cost = dict() if _cost is None else _cost
    _config = dict() if _config is None else _config
    _intra_cache = dict() if _intra_cache is None else _intra_cache
    max_d = k if max_d is None else max_d
    max_t = k if max_t is None else max_t
    if key in _cost: return _cost, _config

    # dp tatble boundary
    if len(nodes) == 0:
        _cost[key], _config[key] = 0, []
        return _cost, _config
    
    assert not (k == 0 or s == 0), \
        f"Illegal configuration: nodes: {len(nodes)} k={k}, s={s}: device number (k) cannot be smaller than pipeline stages (s)"
    assert k >= s, f"Expected k >= s but got k={k}, s={s}"

    # True for 1,2,4,8,16,...
    is_of_power2 = lambda n: (n & (n-1) == 0) and n != 0

    # construct dynamic programming table
    min_val = None  # None means no solution
    for sub1, sub2 in iter_subgraph(nodes, s):
        for d in range(1, min(k + 1, max_d + 1)):
            if mbs % d != 0: continue
            for t in range(1, min(k // d + 1, max_t + 1)):
                # constraints: all devices must be used
                if s == 1 and d * t != k: continue
                # only search for gpu# of power of 2
                if not is_of_power2(t * d): continue
                # guarantee sub-problem searchable
                if k - d * t < s - 1: continue
                # constraints: every device must be used
                if s - 1 > 0 and len(sub2) == 0: continue
                # sub2 cost
                DP(sub2, k-d*t, s-1, intra_solver, mbs, max_d, max_t,
                   _cost, _config, _intra_cache)
                sub2_cost = _cost[(sub2, k-d*t, s-1)]
                if sub2_cost is None: continue
                # sub1 cost: s is also the in-flight microbatch number
                sub1_config = intra_solver(sub1, d, t, s, _cache=_intra_cache)
                if sub1_config is None: continue
                sub1_cost = sub1_config.est_latency
                # pipeline cost
                cost = max(sub1_cost, sub2_cost)
                config = [sub1_config] + _config[(sub2, k-d*t, s-1)]
                # update
                if min_val is None or cost < min_val:
                    min_val = cost
                    _config[(nodes, k, s)] = config

    _cost[key] = min_val
    return _cost, _config


def inter_op(nodes: Tuple[IRFwOperation], ndevs: int, intra_solver: Callable, mbs: int, 
             max_d: Optional[int]=None, max_t: Optional[int]=None, max_p: Optional[int]=None) -> ParallelSpec:
    """
    DP algorithm to search for balanced pipeline stage divisions by considering
    tensor parallelism and pipeline parallelism.

    @param nodes List[IRFwOperation]: graph
    @param ndevs int: number of devices
    @param intra_solver Callable: estimator 
        which takes nodes, tensor parallelism size, data parallelism size
        and in-flight number of microbatches, and outputs of 
        cost (latency in ms) and config (intra-tp config)
    @param mbs: micro-batch size
    @param max_d int: maximal data parallelism size constraint
    @param max_t int: maximal tensor parallelism size constraint

    @return best_config
    """
    nodes: List[IRLayerOp] = cluster_to_layer_ops(nodes)
    nodes = tuple(nodes)
    print(f'> search [search]: constructing dp tables ({len(nodes)} layer ops)...')
    tic = time.time()
    max_d = mbs if max_d is None else max_d
    max_d = min(max_d, mbs, ndevs)
    max_t = ndevs if max_t is None else max_t
    max_t = min(max_t, ndevs)
    max_p = ndevs if max_p is None else min(max_p, ndevs)
    max_p = min(len(nodes), max_p)
    cost, config = None, None
    for nstages in range(1, max_p+1):
        cost, config = DP(nodes, ndevs, nstages, intra_solver, mbs, 
                          max_d, max_t, cost, config)
    print(f'> search [search]: getting optimal results...')
    min_cost, best_config = None, None
    for nstages in range(1, max_p+1):
        tcost = cost[(nodes, ndevs, nstages)]
        if tcost is None: continue
        if min_cost is None or tcost < min_cost:
            min_cost = tcost
            best_config = config[(nodes, ndevs, nstages)]
    assert best_config is not None, f"no solution"
    toc = time.time()
    span = toc - tic
    print(f'> search [finish]: searching time: {span} s')
    print(f'> search [result]: minimal latency per microbatch {min_cost} ms')
    assert all(isinstance(config, StageSpec) for config in best_config)
    spec = ParallelSpec(stages=best_config)
    return spec
