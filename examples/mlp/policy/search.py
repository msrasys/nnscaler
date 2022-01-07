
from typing import Dict, List
from itertools import combinations
from cube.graph import IRGraph
from cube.graph.operator.operator import IRDataOperation, IRFwOperation
import cube.search.iterator as iterator
from cube.profiler.estimator import Estimator

import numpy as np


def get_plan(graph: IRGraph, fnode: IRFwOperation, configs: List[Dict]) -> List[IRFwOperation]:

    all_nodes = [fnode]
    for config in configs:
        sub_nodes = list()
        for node in all_nodes:
            algo = node.algorithms('dim')
            sub = graph.partition(node, algo, config)
            if sub is None:
                sub = graph.replicate(node, times=config['num'])
                fnode.tag = ('rep', 'rep')
            sub_nodes += sub
        all_nodes = sub_nodes
    fnode.tag = tuple('{}-{}'.format(config['name'], config['num']) for config in configs)
    return all_nodes


def compositions(graph: IRGraph, fnode: IRFwOperation, nest: List[int]) -> List[IRFwOperation]:
    """"
    e.g.,
        fnode: linear
        nest: [2, 4]
    will get 9 partition strategies of 8-nodes 
    """
    all_configs = [
        dict(idx=0, dim=0, name='dat'),  # data parallel
        dict(idx=0, dim=1, name='row'),  # row parallel
        dict(idx=1, dim=0, name='col'),  # col parallel
    ]
    config_iter = combinations(all_configs, len(nest))
    for configs in config_iter:
        for config, ndev in zip(configs, nest):
            config['num'] = ndev
        nodes = get_plan(graph, fnode, configs)
        yield nodes
        graph.merge(nodes, fnode)
        fnode.tag = None


def sequence(graph: IRGraph, fnodes: IRFwOperation, resource):

    nest_depth = 2
    nests = iterator.factorization(resource.ngpus, nest_depth)

    if len(fnodes) == 0:
        yield list()

    for fnode in fnodes:
        for nest in nests:
            for seq in compositions(graph, fnode, nest):
                for idx, node in enumerate(seq):
                    graph.assign(node, idx)
                for remain in sequence(graph, fnodes[1:], resource):
                    yield seq + remain


def comm_estimate(graph: IRGraph, ndevice: int) -> int:
    """
    Estimate communications
    """
    estimator = Estimator(graph)
    total_volume = 0
    for devid in range(ndevice):
        total_volume += estimator.comm_volume(devid)
    return total_volume


def PAS(graph: IRGraph, resource):

    # replicate data operation
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)

    # replicate loss operation
    fnodes = [fnode for fnode in graph.nodes() if isinstance(fnode, IRFwOperation)]
    loss = fnodes[-1]
    sub_nodes = graph.replicate(loss, times=resource.ngpus)
    for idx, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, idx)

    # search for linear operations
    fnodes = fnodes[:-1] # only search linears
    seqs = list()
    comms = list()
    plans = list()
    for idx, seq in enumerate(sequence(graph, fnodes, resource)):
        print(f'searching index: {idx}...')
        seqs.append(seq)
        comm = comm_estimate(graph, resource.ngpus)
        comms.append(comm)
        plan = [node.tag for node in fnodes]
        plans.append(plan)
        print(f'comm volume: {comm}')
        # for node in fnodes:
        #     print(node.tag)
        # print(graph.extra_repr())
    print(f'==> grid search done on {idx+1} seq')
    print(f'\n\n')

    comms = np.array(comms)
    indices = np.argsort(comms)

    top_indices = indices[:10]
    top_plan = [plans[idx] for idx in top_indices]
    top_comm = [comms[idx] for idx in top_indices]
    for top_idx, (idx, plan, comm) in enumerate(zip(top_indices, top_plan, top_comm)):
        print(f'top {top_idx} (plan index {idx}):')
        for lid, node in enumerate(plan):
            print(f'linear{lid}: {node}')
        print(f'===> comm: {comm}')

    raise NotImplementedError
