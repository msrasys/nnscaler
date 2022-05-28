from cube.graph.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation


def PAS(graph: IRGraph, resource):
    """
    1F1B scheduling
    """

    num_micro_batch = resource.ngpus
    num_stage = resource.ngpus

    fstages = [list() for _ in range(num_micro_batch * num_stage)]

    def f(micro_batch_id: int, stage_id: int):
        return fstages[micro_batch_id * num_stage + stage_id]

    def b(micro_batch_id: int, stage_id: int):
        fstage = f(micro_batch_id, stage_id)
        bstage = [fnode.mirror for fnode in fstage][::-1]
        return bstage

    fnodes = [node for node in graph.nodes() if isinstance(node, IRFwOperation)]
    stage_op_num = len(fnodes) // num_stage
    for idx, node in enumerate(fnodes):
        stage = min(idx // stage_op_num, num_stage - 1)
        # partition at batch dimension
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(
            node, algo, config=dict(idx=0, dim=0, num=num_micro_batch))
        for mid, sub_node in enumerate(sub_nodes):
            f(mid, stage).append(sub_node)
            graph.assign(sub_node, stage)

    for node in graph.nodes():
        if isinstance(node, IRDataOperation):
            graph.assign(node, 0)
    
    # 1F1B scheduling
    seqs = list()
    # warmup
    for mid in range(num_micro_batch):
        for stage in range(num_stage - mid):
            seqs += f(mid, stage)
    # steady + cooldown:
    for mid in range(num_micro_batch):
        # enqueue backward
        for stage in range(num_stage-1, -1, -1):
            seqs += b(mid, stage)
        # enqueue forward
        for stage in range(num_stage):
            f_mid = mid + num_stage - stage
            if f_mid >= num_micro_batch:
                continue
            seqs += f(f_mid, stage)
    for node in seqs:
        print(node)
    graph.partial_set_order(seqs)

    return graph
