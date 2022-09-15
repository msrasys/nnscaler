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

def PASBranch5(graph: IRGraph, resource):
    '''
    5 way branch
    '''
    assert resource.ngpus == 5

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