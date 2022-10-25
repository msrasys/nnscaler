from typing import List
from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.ir.tensor import IRSubTensor, IRFullTensor

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


def convert_add_to_valmap(graph: IRGraph, add_node: IRFwOperation):
    """
    Remove add node by replacing with tensor valmap
    """
    assert add_node.name == 'add'
    ptensors, producers = [], []
    for itensor in add_node.inputs():
        iptensors = graph.ptensors(itensor.parent)
        assert len(set(t.valmap for t in iptensors)) == len(iptensors)
        ptensors += iptensors
        producers += graph.producers(itensor.parent)
    ftensor = add_node.output(0).parent
    for idx, (ptensor, producer) in enumerate(zip(ptensors, producers)):
        fidx = producer.outputs().index(ptensor)
        bidx = producer.mirror.inputs().index(ptensor.grad)
        ptensor = ftensor.select(ptensor.indmap, (idx, len(producers)))
        ptensor.grad = ftensor.grad.select(ptensor.indmap, (0,1))
        with graph.update(producer):
            producer.set_output(fidx, ptensor)
        with graph.mirror.update(producer.mirror) as bnode:
            bnode.set_input(bidx, ptensor.grad)
    graph.remove(add_node)
    graph.mirror.remove(add_node.mirror)


def flatten_branch_grad(graph: IRGraph, ftensor: IRFullTensor):
    """
    Flatten valmap for different branches.
    """
    assert ftensor.requires_grad
    ctensors = graph.ctensors(ftensor)
    consumers = graph.consumers(ftensor)
    # same tinput ensor
    assert all(ctensor == ctensors[0] for ctensor in ctensors)
    # different gradient (no replicate)
    assert len(set(ctensor.grad.valmap for ctensor in ctensors)) == len(ctensors)
    for idx, (consumer, ctensor) in enumerate(zip(consumers, ctensors)):
        with graph.mirror.update(consumer.mirror) as bnode:
            tidx = bnode.outputs().index(ctensor.grad)
            ctensor.grad = ftensor.grad.select(ctensor.indmap, (idx, len(ctensors)))
            bnode.set_output(tidx, ctensor.grad)


def PASBranch5(graph: IRGraph, resource):
    '''
    5 way branch
    '''
    assert resource.ngpus == 5
    devs = list(range(resource.ngpus))
    for node in graph.select(ntype=IRDataOperation):
        _replica(graph, node, devs)
    for node in graph.select(name='embedding'):
        _tp(graph, node, devs, idx=1, dim=0)
    for node in graph.select(name='linear'):
        _tp(graph, node, devs, idx=1, dim=0)
    for node in graph.select(name='mean'):
        _tp(graph, node, devs, idx=0, dim=2)
    for node in graph.select(name='layernorm'):
        _replica(graph, node, devs)
    for node in graph.select(name='feedforward1'):
        _tp(graph, node, [0, 1], idx=1, dim=1)
    for node in graph.select(name='feedforward2'):
        _tp(graph, node, [2, 3], idx=1, dim=1)
    for node in graph.select(name='feedforward3'):
        _tp(graph, node, [0, 1, 2, 3], idx=2, dim=0)
    for node in graph.select(name='multi_head_attention'):
        graph.assign(node, 4)
    for node in graph.select(name='identity'):
        _replica(graph, node, devs)
    adds = tuple(graph.select(name='add'))
    assert len(adds) == 2
    # graph.assign(adds[0], 4)
    convert_add_to_valmap(graph, adds[0])
    _replica(graph, adds[1], devs)
    # convert_add_to_valmap(graph, adds[1])
    for node in graph.select('feedforward1'):
        ftensor = node.input(0).parent
        break
    flatten_branch_grad(graph, ftensor)
    print(graph.extra_repr())
    return graph