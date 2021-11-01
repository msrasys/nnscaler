from cube.schedule.su import SUType

def choose_input(op, input_incarnation): pass
def choose_output(op, output_incarnation): pass
def create_incar(graph, tensor_or_op): pass


def transformation_policy(graph, resource):

    def _recompute_ops(graph, ops):
        """
        PyTorch Checkpointing
        """
        tensors_incar = list()
        ops_incar = list()

        for op in ops[:-1]:
            op_incar = graph.create_incar(op)
            ops_incar.append(op_incar)
            for output in op.outputs():
                tensor_incar = graph.create_incar(output)
                tensors_incar.append(tensor_incar)
                ops_incar.choose_output(tensor_incar)
        for op in ops_incar[1:]:
            for input in op.outputs():
                for input_incar in input.get_incar():
                    if input_incar in tensors_incar:
                        graph.choose_input(op, input_incar)
                    # else keep in memory
        for op in ops[1:]:
            for idx, output in enumerate(op.outputs()):
                succ_ops = graph.successors(op, idx)
                succ_ops = [
                    op for op in succ_ops if op.type == SUType.Backward
                ]
                for succ_op in succ_ops:
                    for input in succ_op.inputs():
                        for input_incar in input.get_incar():
                            if input_incar in tensors_incar:
                                graph.choose_input(succ_op, input_incar)

    # checkpointing tensor
    chunk_num = 4
    # forward ops
    fops = [node for node in graph.nodes() if node.type == SUType.Forward]
    chunk_size = int(len(fops) // chunk_num)
    for cid in range(chunk_num):
        chunk_fops = fops[chunk_size * cid, chunk_size * (cid + 1)]
        _recompute_ops(graph, chunk_fops)


def schedule_policy(sugraph, resource):

    for su in sugraph.sus():
        sugraph.assign(su, 0)
        if su.is_incarnation():
            succ_sus = sugraph.successors(su)
            for succ_su in succ_sus:
                if sugraph.merge(su, succ_su):
                    break
    sugraph.set_order(sugraph.random_topo_order())
