
from typing import List, Tuple, Dict, Optional
import multiprocessing
import numpy as np
import warnings
import time

from nnscaler.ir.cten import IRTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.function.anchor import IRGraphAnchor

from examples.policies.alpa.layer_op import IRLayerOp
from examples.policies.alpa.cost_model import CostModel
from examples.policies.alpa.plan import StageSpec

# ILP solver
import pulp
from pulp import LpVariable, LpProblem, LpMinimize, LpStatus, lpSum, lpDot, LpStatus


def intra_op(layer_nodes: List[IRLayerOp], dp_size: int, tp_size: int,
             inflights: int, recompute: bool, memory_limit: int,
             cost_model: CostModel, _cache: Dict = None) -> Optional[StageSpec]:
    """
    Search for the best intra-op parallelism configuration given device mesh.
    The search is only suitable for training.
    """
    key = (layer_nodes, dp_size, tp_size)
    if isinstance(_cache, dict) and key in _cache: return _cache[key]

    tic = time.time()

    fnodes: List[IRFwOperation] = []
    for layer_op in layer_nodes:
        for node in layer_op.nodes:
            if isinstance(node, IRGraphAnchor) or node.name == 'multiref': continue
            fnodes.append(node)

    # search for tp configuration

    # create variables (nodes)
    s, d, c = {}, {}, {}  # partition index, computation cost, communication cost
    e, r = [], []  # inter-node resharding cost

    num_nodes = 0
    for fnode in fnodes:
        cid = fnode.cid
        npartitions = len(cost_model.partition_algos[fnode.cid])
        s[cid] = LpVariable.matrix(f's[{num_nodes}]', (range(npartitions),), cat='Binary')
        d[cid] = cost_model.get_comp_cost(fnode, tp_size).flatten() / dp_size
        c[cid] = cost_model.get_comm_cost(fnode, tp_size).flatten() / dp_size
        # setup initial value
        for pidx, strategy in enumerate(cost_model.partition_algos[fnode.cid]):
            if strategy is None: continue
            idx, dim = strategy
            identifier = fnode.anno.input(idx)[dim].identifiers[0]
            if fnode.anno.getlen(identifier) % (tp_size * dp_size) != 0:
                # print(f'remove transform choice on {fnode.name}({fnode.cid}) '
                #       f'of strategy: {strategy} for tp={tp_size}, dp={dp_size}')
                s[cid][pidx].setInitialValue(False)
                s[cid][pidx].fixValue()
        num_nodes += 1

    edges = cost_model.get_edges(fnodes)
    num_edges = 0
    for src, dsts in edges.items():
        for dst in dsts:
            nsrc = len(cost_model.partition_algos[src.cid])
            ndst = len(cost_model.partition_algos[dst.cid])
            e.append(LpVariable.matrix(f"e[{src.cid}, {dst.cid}]",
                                       (range(nsrc * ndst),),
                                       cat='Binary'))
            r.append(cost_model.get_pair_reshard_cost(src, dst, tp_size).flatten())
            num_edges += 1

    # initial value: --skip

    # objective
    prob = LpProblem('intra_op', LpMinimize)
    # computation cost
    obj = 0
    for fnode in fnodes:
        cid = fnode.cid
        obj += lpDot(s[cid], c[cid]) + lpDot(s[cid], d[cid])
    # communication cost
    for i in range(num_edges):
        obj += lpDot(e[i], r[i])
    
    prob += obj
    
    # constraints

    # a) only one partition can be selected
    for fnode in fnodes:
        prob += lpSum(s[fnode.cid]) == 1
    for i in range(num_edges):
        prob += lpSum(e[i]) == 1

    # e_src_dst[i][j] = 1 => s_src[i] == 1 and s_dst[j] == 1
    eidx = 0
    for src, dsts in edges.items():
        for dst in dsts:
            for row in range(len(s[src.cid])):
                C = len(s[dst.cid])
                prob += lpSum(
                    e[eidx][row * C + col] for col in range(0, C)) <= s[src.cid][row]
            for col in range(len(s[dst.cid])):
                R = len(s[src.cid])
                C = len(s[dst.cid])
                prob += lpSum(
                    e[eidx][row * C + col] for row in range(0, R)) <= s[dst.cid][col]
            eidx += 1

    # b) memory constraint --skip
    
    assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True), (
        "Please install ILP solvers by 'sudo apt install coinor-cbc' or 'pip install pulp'")
    
    time_limit = 600
    solver = pulp.PULP_CBC_CMD(
        mip=True, msg=0, 
        timeLimit=time_limit, 
        threads=multiprocessing.cpu_count())
    prob.solve(solver)

    status = prob.status
    objective = pulp.value(prob.objective)
    objective = float(objective) if objective is not None else -1.0
    # print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}")
    # print(f"#nodes: {num_nodes},  #edges: {num_edges}")
    # print(f'ILP search time: {time.time() - tic:.2f} seconds')

    # reshard_cost = 0
    # for i in range(num_edges):
    #     reshard_cost += lpDot(e[i], r[i])
    # reshard_cost = pulp.value(reshard_cost)
    # print(f'debug info: reshard cost: {reshard_cost}')

    if prob.status in [pulp.LpStatusInfeasible]:
        raise RuntimeError("Cannot run the function under the given memory budget.")
    
    def get_non_zero_index(binary_vector):
        """Get the index of non-zero item in a vector."""
        ct = 0
        ret = None
        for i, elem in enumerate(binary_vector):
            if pulp.value(elem):
                ret = i
                ct += 1

        assert ct == 1
        return ret

    tp_spec: Dict[int, int] = {}
    for fnode in fnodes:
        index = get_non_zero_index(s[fnode.cid])
        tp_spec[fnode.cid] = index

    # check results
    e_val = np.full((num_edges,), -1, dtype=np.int32)
    eidx = 0
    for (src, dsts) in edges.items():
        for dst in dsts:
            e_val[eidx] = get_non_zero_index(e[eidx])
            src_spec_index = e_val[eidx] // len(s[dst.cid])
            dst_spec_index = e_val[eidx] % len(s[dst.cid])
            assert src_spec_index == tp_spec[src.cid]
            assert dst_spec_index == tp_spec[dst.cid]
            eidx += 1
    
    if objective > 1e13:
        warnings.warn("Detect unexpected behaviors in the auto-sharding pass.")
    
    # estimate activation memory
    non_recompute_mem = 0
    recompute_mem, curr_recomp_id = [0], None
    for node in fnodes:
        strat = cost_model.partition_algos[node.cid][tp_spec[node.cid]]
        op_tp_size = 1 if strat is None else tp_size
        node_mem = cost_model.get_memory_cost(node) // (dp_size * op_tp_size)
        if node.recompute != curr_recomp_id:
            recompute_mem.append(0)
            curr_recomp_id = node.recompute
        if node.recompute is None:
            non_recompute_mem += node_mem
        else:
            recompute_mem[-1] += node_mem
    act_memory = non_recompute_mem * inflights + max(recompute_mem)

    # estimate parameter memory
    param_mem = 0
    pids = set()
    for node in fnodes:
        attrs = [t for t in node.inputs() if \
                 isinstance(t, IRTensor) and t.is_attr()]
        for attr in attrs:
            if attr.tid in pids: continue
            opt = 4 if attr.is_param() else 1
            # we estimate parameter size by assuming it will partition on weight
            param_mem += opt * attr.byte_size() // tp_size
            pids.add(attr.tid)

    # print(f'debug: inflights: {inflights}, act memory: {act_memory/1024/1024/1024}, param mem: {param_mem/1024/1024/1024}')
    mem_cost = act_memory + param_mem
    if mem_cost > memory_limit:
        print(f'searching results of {len(tp_spec)} nodes: tp={tp_size}, dp={dp_size}: no solution (memory: {mem_cost/1024/1024/1024} GB)')
        return None

    # get tensor parallelism spec
    stage_tp_spec = {}
    names = {}
    for fnode in fnodes:
        strategy = None if tp_size == 1 else \
                   cost_model.partition_algos[fnode.cid][tp_spec[fnode.cid]]
        stage_tp_spec[fnode.cid] = strategy
        names[fnode.cid] = fnode.name

    config = StageSpec(
        est_latency=objective / 3 * 4 if recompute else objective,
        est_memory=mem_cost,
        tp_size=tp_size,
        dp_size=dp_size,
        tp_spec=stage_tp_spec,
        names=names,
    )
    print(f'searching results of {len(stage_tp_spec)} nodes: tp={tp_size}, dp={dp_size} '
          f'latency={objective}, memory={mem_cost/1024/1024/1024} GB')
    if isinstance(_cache, dict): _cache[key] = config
    # print(config)
    return config
