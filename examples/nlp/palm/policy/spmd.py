from typing import List
from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation

def PASSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1

    for node in graph.nodes():
        if isinstance(node, (IRDataOperation, IRFwOperation)):
            graph.assign(node, 0)

    return graph


def PASData(graph: IRGraph, resource):
    '''
    2 way Data Parallel
    '''
    # assert resource.ngpus == 2

    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, num=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
            batch_dim = node.get_batch_dims()[0]

    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(node,
                                        algo,
                                        idx=0,
                                        dim=batch_dim,
                                        num=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
    return graph


def PASMegatron(graph: IRGraph, resource):
    tp_size = resource.ngpus
    tp_devs = list(range(tp_size))

    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, num=resource.ngpus)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
            batch_dim = node.get_batch_dims()[0]

    def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int], idx: int, dim: int):
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
        assert sub_nodes is not None
        for devid, sub_node in zip(devs, sub_nodes):
            graph.assign(sub_node, devid)
        return sub_nodes

    def _replica(graph: IRGraph, node: IRFwOperation, devs: List[int]):
        sub_nodes = graph.replicate(node, times=len(devs))
        for dev_id, sub_node in zip(devs, sub_nodes):
            graph.assign(sub_node, dev_id)
        return sub_nodes
    
    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            if node.name == 'embedding':
                _tp(graph, node, tp_devs, idx=1, dim=0)
            elif node.name == "linear":
                _tp(graph, node, tp_devs, idx=1, dim=0)
            elif node.name == 'multi_head_attention':
                # TODO: data parallel current
                algo = node.algorithms('dim')
                sub_nodes = graph.partition(node,
                                            algo,
                                            idx=0,
                                            dim=batch_dim,
                                            num=resource.ngpus)
                for idx, sub_node in enumerate(sub_nodes):
                    graph.assign(sub_node, idx)
            elif node.name == 'feedforward1':
                _tp(graph, node, tp_devs, idx=1, dim=1)
            elif node.name == 'feedforward2':
                _tp(graph, node, tp_devs, idx=1, dim=1)
            elif node.name == 'feedforward3':
                _tp(graph, node, tp_devs, idx=2, dim=0)
            elif node.name == 'mean':
                _tp(graph, node, tp_devs, idx=0, dim=2)
            else:
                _replica(graph, node, tp_devs)
    return graph