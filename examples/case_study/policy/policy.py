
def select(tensor, indices, val_map_op=None, shape=None): pass

def input_adapter(inputs, target): pass

def iter_op(DAG): pass
def generate_for_each_rank(pDAG): pass


def sschedule_dp(pDAG, resources, input_tensors):
    """
    Data Parallel Description

    Args:
        * pDAG: partial semantic (logical) computation graph
        * Resources: Environment inlcuding devices, network topology etc
    Returns:
        * pDAGs (list[DAG]) execution (local & physical) DAG for each rank
    """
    ndevs = resources.ndevs
    for data in input_tensors:
        shape = data.shape
        for sid in range(ndevs):
            chunk_shape = ()
            for dim, size in enumerate(shape):
                if dim == 0:
                    chunk_size = shape[0] // ndevs
                    chunk_shape.append(slice(sid * chunk_size, (sid+1) * chunk_size))
                else:
                    chunk_shape.append(slice(0, size))
            data.add_segment(select(data, chunk_shape, None))
    pDAG.op[0].set_partition(input_tensors)
    for op in iter_op(pDAG):
        # inputs: op input tensors
        # outputs: op output tensors
        for inputs, dist_op, outputs in op.dist_candidates():
            # find the data parallelism
            if dist_op.satisfy(inputs):
                # set placement
                dist_op.op_placement = list(range(ndevs))
                # replace logical op to data parallelism
                input_adapter(dist_op.inputs, target=inputs)
                # output will be in data parallel format
                pDAG.replace(op, dist_op)
    # materialize to physical op
    DAGs = generate_for_each_rank(pDAG, resources)
    return DAGs
