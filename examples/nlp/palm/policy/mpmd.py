from typing import List
from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation

def PASBranch3(graph: IRGraph, resource):
    '''
    3 way branch
    '''
    assert resource.ngpus == 3

    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, num=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
            batch_dim = node.get_batch_dims()[0]

    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            if node.name == 'embedding' or node.name == 'linear':
                # data parallel
                algo = node.algorithms('dim')
                sub_nodes = graph.partition(node,
                                            algo,
                                            idx=0,
                                            dim=batch_dim,
                                            num=resource.ngpus)
                for idx, sub_node in enumerate(sub_nodes):
                    graph.assign(sub_node, idx)
            elif node.name == 'layernorm' or node.name == 'multiref' or node.name == 'add' or node.name == 'mean':
                # replicate
                sub_nodes = graph.replicate(node, times=resource.ngpus)
                for idx, sub_node in enumerate(sub_nodes):
                    graph.assign(sub_node, idx)
            elif node.name == 'feedforward1':
                graph.assign(node, 0)
            elif node.name == 'feedforward2':
                graph.assign(node, 1)
            elif node.name == 'feedforward3':
                algo = node.algorithms('dim')
                sub_nodes = graph.partition(node, algo, idx=2, dim=0, num=2)
                graph.assign(sub_nodes[0], 0)
                graph.assign(sub_nodes[1], 1)
            elif node.name == 'multi_head_attention':
                graph.assign(node, 2)
            else:
                assert False, node.name

    return graph


def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int], idx: int, dim: int):
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
    assert sub_nodes is not None
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def _replica(graph: IRGraph, node: IRFwOperation, devs: List[int]):
    sub_nodes = graph.replicate(node, times=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def PASBranch5(graph: IRGraph, resource):
    '''
    5 way branch
    '''
    assert resource.ngpus == 5

    devs = list(range(resource.ngpus))

    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            _replica(graph, node, devs)

    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            if node.name == 'embedding':
                _tp(graph, node, devs, idx=1, dim=0)
            elif node.name == 'linear':
                _tp(graph, node, devs, idx=1, dim=0)
            elif node.name == 'mean':
                _tp(graph, node, devs, idx=0, dim=2)
            elif node.name == 'layernorm' or node.name == 'multiref' or node.name == 'add':
                _replica(graph, node, devs)
            elif node.name == 'feedforward1':
                algo = node.algorithms('dim')
                sub_nodes = graph.partition(node, algo, idx=1, dim=1, num=2)
                graph.assign(sub_nodes[0], 0)
                graph.assign(sub_nodes[1], 1)
            elif node.name == 'feedforward2':
                algo = node.algorithms('dim')
                sub_nodes = graph.partition(node, algo, idx=1, dim=1, num=2)
                graph.assign(sub_nodes[0], 2)
                graph.assign(sub_nodes[1], 3)
            elif node.name == 'feedforward3':
                algo = node.algorithms('dim')
                sub_nodes = graph.partition(node, algo, idx=2, dim=0, num=4)
                graph.assign(sub_nodes[0], 0)
                graph.assign(sub_nodes[1], 1)
                graph.assign(sub_nodes[2], 2)
                graph.assign(sub_nodes[3], 3)
            elif node.name == 'multi_head_attention':
                graph.assign(node, 4)
            else:
                assert False, node.name

    return graph
