from cube.graph import IRGraph
from cube.ir.operator import IRFwOperation, IRDataOperation


def PAS(graph: IRGraph, resource):

    assert resource.ngpus == 4, "the optimal plan is for 4 GPU case."

    # replicate data operation
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            sub_nodes = graph.replicate(node, times=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)

    # replicate loss operation
    fnodes = [fnode for fnode in graph.nodes() if isinstance(fnode, IRFwOperation)]
    loss = fnodes[-1]
    sub_nodes = graph.replicate(loss, times=resource.ngpus)
    for idx, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, idx)

    fnodes = fnodes[:-1]
    # linear0 config
    config0 = [
        None,
        dict(idx=1, dim=0, num=4) # col
    ]
    # linear1 config
    config1 = [
        dict(idx=0, dim=1, num=2), # row
        dict(idx=1, dim=0, num=2), # col
    ]
    # linear2 config
    config2 = [
        dict(idx=0, dim=0, num=2), # dat
        dict(idx=0, dim=1, num=2), # row
    ]
    # linear3 config
    config3 = [
        dict(idx=0, dim=0, num=2), # dat
        dict(idx=0, dim=1, num=2), # row
    ]
    configs = [config0, config1, config2, config3]
    assert len(fnodes) == len(configs)
    for fnode, config in zip(fnodes, configs):
        all_nodes = [fnode]
        for conf in config:
            if conf is None:
                continue
            sub_nodes = list()
            for node in all_nodes:
                algo = node.algorithms('dim')
                nodes = graph.partition(node, algo, conf)
                sub_nodes += nodes
            all_nodes = sub_nodes
        assert len(all_nodes) == 4
        for idx, node in enumerate(all_nodes):
            graph.assign(node, idx)
    return graph
