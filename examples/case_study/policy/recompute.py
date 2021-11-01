from cube.schedule.su import SUType


def transformation_policy(graph, resource):

    def _recompute_op(graph, ops):
        """
        PyTorch Checkpointing
        """
        for op in ops[1:-1]:
            for idx, output in enumerate(op.outputs()):
                succ_ops = graph.successors(op, idx)
                succ_ops = [
                    op for op in succ_ops if op.type == SUType.Backward
                ]
                # remove output tensor connection between op -> [succ_ops],
                # duplicate op with to connect with succ_ops
                graph.incarnation(output, op, succ_ops)

    # checkpointing tensor
    chunk_num = 4
    # forward ops
    fops = [node for node in graph.nodes() if node.type == SUType.Forward]
    chunk_size = int(len(fops) // chunk_num)
    for cid in range(chunk_num):
        chunk_fops = fops[chunk_size * cid, chunk_size * (cid + 1)]
        _recompute_op(graph, chunk_fops)


def schedule_policy(sugraph, resource):

    for su in sugraph.sus():
        sugraph.assign(su, 0)
        if su.is_incarnation():
            succ_sus = sugraph.successors(su)
            for succ_su in succ_sus:
                if sugraph.merge(su, succ_su):
                    break
    sugraph.set_order(sugraph.random_topo_order())
