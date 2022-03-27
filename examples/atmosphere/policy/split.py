from cube.graph import IRGraph
from cube.graph.adapter.adapter import IRAdapter
from cube.graph.operator.operator import IRBpOperation, IRDataOperation, IRFwOperation
from cube.graph.operator.function.conv import IRConv3D
from cube.graph.operator.function.pad import IRPad

def PAS(graph: IRGraph, resource):
    print(graph.extra_repr())
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            if isinstance(node, IRDataOperation):
                print(f'### IRDataOperation = {node}')
                algo = node.algorithms('data')
                sub_nodes = graph.partition(node, algo, config=dict(num=resource.ngpus))
                for idx, subnode in enumerate(sub_nodes):
                    graph.assign(subnode, idx)
            elif isinstance(node, IRPad):
                print(f'### IRPad = {node}')
                sub_nodes = list()
                algo = node.algorithms('dim')
                sub_nodes = graph.partition(node, algo, config=dict(dim=1, num=min(2, resource.ngpus)))
                for idx, sub_node in enumerate(sub_nodes):
                    graph.assign(sub_node, idx)
            elif isinstance(node, IRConv3D):
                print(f'### IRConv3D = {node}')
                sub_nodes = list()
                algo = node.algorithms('halo')
                Wnodes = graph.partition(node, algo, config=dict(idx=0, dim=3, num=resource.ngpus // 2))
                for Wnode in Wnodes:
                    algo = Wnode.algorithms('halo')
                    Hnodes = graph.partition(Wnode, algo, config=dict(idx=0, dim=2, num=2))
                    sub_nodes += Hnodes
            else:
                print(f'### to-replicate = {node}')
                sub_nodes = graph.replicate(node, times=resource.ngpus, reset_dependency=False)
                for idx, node in enumerate(sub_nodes):
                    graph.assign(node, idx)
        else:
            print(f'### non-IRBpOperation = {node}')
    return graph
