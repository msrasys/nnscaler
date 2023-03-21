from typing import List
from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.graph.gener.rvd.intra import IntraAutoPlacer
from cube.graph.function import IRTensor


# tensor parallelism with auto-placer
# This is an implementation example of SPMD auto placer usage
def _tp_autoplace(graph: IRGraph, node: IRFwOperation, devs: List[int], **configs):

    if len(devs) == 1:
        graph.assign(node, devs[0])
        return [node]

    segment: IRSegment = graph.segment(node)
    ftensor = node.input(configs['idx']).parent

    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, **configs)
    assert sub_nodes is not None
    producers = segment.producers(ftensor)
    if ftensor.is_param() or len(producers) != len(sub_nodes):
        print(f"> skip auto placer due to condition not matched: "
              f"nproducers: {len(producers)}, nconsumers: {len(sub_nodes)}, "              f"producer name: {producers[0].name if len(producers) > 0 else None}")
        devs = sorted(list(devs))
        for devid, node in zip(devs, sub_nodes):
            graph.assign(node, devid)
    else:
        devices = IntraAutoPlacer.auto_place(
            segment, ftensor, producers, sub_nodes)
        for devid, subnode in zip(devices, sub_nodes):
            graph.assign(subnode, devid)
    return sub_nodes

# tensor parallelism
def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int], **configs):
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, **configs)
    assert sub_nodes is not None
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes

# replicate
def _replica(graph: IRGraph, node: IRFwOperation, devs: List[int]):
    sub_nodes = graph.replicate(node, times=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


def PASSingle(graph: IRGraph, resource):
    """
    Single device
    """
    assert resource.ngpus == 1, "only apply for single gpu case"
    for node in graph.nodes():
        if isinstance(node, (IRDataOperation, IRFwOperation)):
            graph.assign(node, 0)
    return graph


def PASData(graph: IRGraph, resource):
    """
    Data Parallel
    """
    # auto multi-ref
    for ftensor in graph.full_tensors():
        if len(graph.consumers(ftensor)) > 1:
            if ftensor.is_attr():
                continue
        graph.multiref(ftensor, [[n] for n in graph.consumers(ftensor)])

    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            algo = node.algorithms('data')
            sub_nodes = graph.partition(node, algo, num=resource.ngpus)
            for idx, subnode in enumerate(sub_nodes):
                graph.assign(subnode, idx)
            batch_dim = node.get_batch_dims()[0]
    for node in graph.nodes():
        if isinstance(node, IRFwOperation):
            try:
                algo = node.algorithms('dim')
                idx = 0
                sub_nodes = graph.partition(
                    node, algo, idx=idx, dim=batch_dim, num=resource.ngpus)
            except AssertionError:
                print(f'WARNING: {node} cannot find dim algo, using replicate instead')
                sub_nodes = graph.replicate(node, resource.ngpus)

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
    devs = list(range(resource.ngpus))

    for dl in graph.select(ntype=IRDataOperation):
        sub_nodes = graph.replicate(dl, resource.ngpus)
        for idx, node in enumerate(sub_nodes):
            graph.assign(node, idx)

    for node in graph.select(ntype=IRFwOperation):
        if node.name == 'linear':
            _tp(graph, node, devs, idx=0, dim=1, num=len(devs))
        else:
            _replica(graph, node, devs)

    return graph


def PASHybrid(graph: IRGraph, resource):
    """
    Linear Hybrid Parallelism (Megatron)
    """
    linears = [node for node in graph.nodes() if node.name == 'linear']
    for idx, node in enumerate(linears):
        try:
            algo = node.algorithms('dim')
            tp_nodes = graph.partition(node, algo, idx=1, dim=idx%2, num=resource.ngpus)
            for idx, node in enumerate(tp_nodes):
                graph.assign(node, idx)
        except AssertionError:
            print(f'WARNING: {node} cannot find dim algo, using replicate instead')
            sub_nodes = graph.replicate(node, resource.ngpus)
            for idx, node in enumerate(sub_nodes):
                graph.assign(node, idx)

    for node in graph.nodes():
        if isinstance(node, (IRFwOperation, IRDataOperation)):
            if len(node.device) == 0:
                sub_nodes = graph.replicate(node, resource.ngpus)
                for idx, node in enumerate(sub_nodes):
                    graph.assign(node, idx)
    print(graph.extra_repr())
    return graph


def PASMegatronTP(graph: IRGraph, resource):
    """
    Tensor + Data Parallelism
    """
    tp = min(2, resource.ngpus)
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

