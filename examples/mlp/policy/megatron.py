from cube.graph import IRGraph
from cube.graph.operator.operator import IRFwOperation, IRDataOperation


def PAS(graph: IRGraph, resource):
    """
    Linear Hybrid + Nested Partition
    """
    tp = 4
    dp = resource.ngpus // tp
    for idx, node in enumerate(graph.nodes()):
        if isinstance(node, IRDataOperation):
            sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
            continue
        if isinstance(node, IRFwOperation):
            sub_nodes = list()
            algo = node.algorithms('dim')
            tp_nodes = graph.partition(
                node, algo, config=dict(idx=1, dim=(idx+1)%2, num=tp)
            )
            if tp_nodes is not None:
                for tp_node in tp_nodes:
                    algo = tp_node.algorithms('dim')
                    dp_nodes = graph.partition(tp_node, algo, config=dict(idx=0, dim=0, num=dp))
                    sub_nodes += dp_nodes
            else:
                sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    # print(graph.extra_repr())
    return graph
