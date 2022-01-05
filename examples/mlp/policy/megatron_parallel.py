from cube.graph import IRGraph
from cube.graph.operator.operator import IRFwOperation, IRDataOperation


def PAS(graph: IRGraph, resource):
    """
    Linear Hybrid + Nested Partition
    """
    tp = 2
    dp = resource.ngpus // tp
    for idx, node in enumerate(graph.nodes()):
        if isinstance(node, IRFwOperation) or isinstance(node, IRDataOperation):
            algo = node.algorithms('dim')
            if algo:
                sub_nodes = list()
                tp_nodes = graph.partition(
                    node, algo,
                    config=dict(idx=1, dim=(idx+1)%2, num=tp)
                )
                for tp_node in tp_nodes:
                    algo = tp_node.algorithms('dim')
                    dp_nodes = graph.partition(
                        tp_node, algo,
                        config=dict(idx=0, dim=0, num=dp)
                    )
                    sub_nodes += dp_nodes
            else:
                sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    print(graph.extra_repr())
    return graph
