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
            if idx % 2 == 0:
                algo = node.algorithms('row')
            else:
                algo = node.algorithms('column')
            if algo:
                sub_nodes = list()
                tp_nodes = graph.partition(
                    node, algo, config=dict(chunk_num=tp)
                )
                for tp_node in tp_nodes:
                    algo = tp_node.algorithms('data')
                    dp_nodes = graph.partition(
                        tp_node, algo, config=dict(chunk_num=dp))
                    sub_nodes += dp_nodes
            else:
                sub_nodes = [node]
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    print(graph.extra_repr())
    return graph
