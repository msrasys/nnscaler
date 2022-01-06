
from typing import Dict, List
from itertools import combinations
from cube.graph import IRGraph
from cube.graph.operator.operator import IRDataOperation, IRFwOperation
import cube.search.iterator as iterator


def get_plan(graph: IRGraph, fnode: IRFwOperation, configs: List[Dict]) -> List[IRFwOperation]:

    all_nodes = [fnode]
    for config in configs:
        sub_nodes = list()
        for node in all_nodes:
            algo = node.algorithms('dim')
            sub = graph.partition(node, algo, config)
            if sub is None:
                sub = graph.replicate(node, times=config['num'])
            sub_nodes += sub
        all_nodes = sub_nodes
    return all_nodes


def compositions(graph: IRGraph, fnode: IRFwOperation, nest: List[int]) -> List[IRFwOperation]:
    """"
    e.g.,
        fnode: linear
        nest: [2, 4]
    will get 9 partition strategies of 8-nodes 
    """
    all_configs = [
        dict(idx=0, dim=0),  # data parallel
        dict(idx=0, dim=1),  # row parallel
        dict(idx=1, dim=0),  # col parallel
    ]
    config_iter = combinations(all_configs, len(nest))
    for configs in config_iter:
        for config, ndev in zip(configs, nest):
            config['num'] = ndev
        nodes = get_plan(graph, fnode, configs)
        yield nodes
        graph.merge(nodes, fnode)


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
            

def PAS(graph: IRGraph, resource):

    # replicate data operation
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)

    fnodes = [fnode for fnode in graph.nodes() if isinstance(fnode, IRFwOperation)]

    for idx, seq in enumerate(sequence(graph, fnodes, resource)):
        print(f'searching index: {idx}')
        print(graph.extra_repr())
        # for node in seq:
        #     print(node)
        # print('\n')
    print(f'==> grid searched on {idx+1} seq')

    raise NotImplementedError
