import math
import random

from cube.ir.operator import IRDataOperation, IRFwOperation


def PAS(graph, resource):
    """
    Random pipeline
    """
    micro_batch_num = resource.ngpus
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            device = random.randint(0, resource.ngpus - 1)
            graph.assign(node, device)
        if isinstance(node, IRFwOperation):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(
                node, algo, config=dict(idx=0, dim=0, num=micro_batch_num))
            if sub_nodes is None:
                sub_nodes = [node]
            for idx, sub_node in enumerate(sub_nodes):
                device = random.randint(0, resource.ngpus - 1)
                graph.assign(sub_node, device)
    print(graph.extra_repr())
    return graph
