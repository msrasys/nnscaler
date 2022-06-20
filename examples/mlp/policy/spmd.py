from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation


def PASSingle(graph: IRGraph, resource):
    """
    Single device
    """
    assert resource.ngpus == 1, "only apply for single gpu case"
    for node in graph.nodes():
        graph.assign(node, 0)
    return graph


def PASData(graph: IRGraph, resource):
    """
    Data Parallel
    """
    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, num=resource.ngpus)
            for idx, subnode in enumerate(sub_nodes):
                graph.assign(subnode, idx)
            batch_dim = node.get_batch_dims()[0]
    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            algo = node.algorithms('dim')
            sub_nodes = graph.partition(
                node, algo, idx=0, dim=batch_dim, num=resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)
    return graph


def PASCol(graph: IRGraph, resource):
    """
    Linear Column Parallel
    """
    linears = [node for node in graph.nodes() if node.name == 'linear']
    for idx, node in enumerate(linears):
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(
            node, algo, idx=1, dim=0, num=resource.ngpus
        )
        for idx, node in enumerate(sub_nodes):
            graph.assign(node, idx)
    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)):
            if len(node.device) == 0:
                sub_nodes = graph.replicate(node, resource.ngpus)
                for idx, node in enumerate(sub_nodes):
                    graph.assign(node, idx)
    return graph


def PASRow(graph: IRGraph, resource):
    """
    Linear Column Parallel
    """
    linears = [node for node in graph.nodes() if node.name == 'linear']
    for idx, node in enumerate(linears):
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(
            node, algo, idx=1, dim=1, num=resource.ngpus
        )
        for idx, node in enumerate(sub_nodes):
            graph.assign(node, idx)
    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)):
            if len(node.device) == 0:
                sub_nodes = graph.replicate(node, resource.ngpus)
                for idx, node in enumerate(sub_nodes):
                    graph.assign(node, idx)
    return graph


def PASHybrid(graph: IRGraph, resource):
    """
    Linear Hybrid Parallelism (Megatron)
    """
    linears = [node for node in graph.nodes() if node.name == 'linear']
    for idx, node in enumerate(linears):
        algo = node.algorithms('dim')
        tp_nodes = graph.partition(node, algo, idx=1, dim=idx%2, num=resource.ngpus)
        for idx, node in enumerate(tp_nodes):
            graph.assign(node, idx)
    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)):
            if len(node.device) == 0:
                sub_nodes = graph.replicate(node, resource.ngpus)
                for idx, node in enumerate(sub_nodes):
                    graph.assign(node, idx)
    print(graph.extra_repr())
    return graph


def PASMegatron(graph: IRGraph, resource):
    """
    Tensor + Data Parallelism
    """
    tp = 2
    dp = resource.ngpus // tp
    linears = [node for node in graph.nodes() if node.name == 'linear']
    for idx, node in enumerate(linears):
        sub_nodes = []
        algo = node.algorithms('dim')
        tp_nodes = graph.partition(node, algo, idx=1, dim=idx%2, num=tp)
        for tp_node in tp_nodes:
            algo = tp_node.algorithms('dim')
            dp_nodes = graph.partition(tp_node, algo, idx=0, dim=0, num=dp)
            sub_nodes += dp_nodes
        for idx, node in enumerate(sub_nodes):
            graph.assign(node, idx)
    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)):
            if len(node.device) == 0:
                sub_nodes = graph.replicate(node, resource.ngpus)
                for idx, node in enumerate(sub_nodes):
                    graph.assign(node, idx)
    # print(graph.extra_repr())
    return graph


def PASOptimal(graph: IRGraph, resource):
    """
    Square Linear optimal parallelism (4GPU)
    """
    assert resource.ngpus == 4, "only apply to 4 GPU case"

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
                nodes = graph.partition(node, algo, **conf)
                sub_nodes += nodes
            all_nodes = sub_nodes
        assert len(all_nodes) == 4
        for idx, node in enumerate(all_nodes):
            graph.assign(node, idx)
    return graph

