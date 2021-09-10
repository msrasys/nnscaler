import torch
 
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
        # set num micro-batch to 4
        for sid in range(ndevs * 4):
            chunk_shape = ()
            for dim, size in enumerate(shape):
                if dim == 0:
                    chunk_size = shape[0] // ndevs // 4
                    chunk_shape.append(slice(sid * chunk_size, (sid+1) * chunk_size))
                else:
                    chunk_shape.append(slice(0, size))
            data.add_segment(select(data, chunk_shape, None))
    pDAG.op[0].set_partition(input_tensors)
    for inputs, op, outputs in iter_op(pDAG):
        # inputs: op input tensors
        # outputs: op output tensors
        for dist_op in op.dist_candidates():
            # find the data parallelism
            if dist_op.satisfy_and_set(inputs):
                # set placement
                dist_op.op_placement = list(range(ndevs))
                # replace logical op to data parallelism
                input_adapter(dist_op.inputs, target=inputs)
                # output will be in data parallel format
                pDAG.replace(op, dist_op)
    # materialize to physical op
    DAGs = generate_for_each_rank(pDAG, resources)
    return DAGs


def tschedule_1f1b(actions, relations, resources):
    """
    Pipeline 1f1b policy description -- each device order

    Actions: a list of actions

    relations: list[(Action1, Action2)]: a list of tuples indicate partial order
    """
    num_stage = resources.n_gpus
    num_microbatch = len(actions) / 2 / num_stage

    f = lambda stage, micro_batch_id: actions[2 * micro_batch_id * num_stage + stage]
    b = lambda stage, micro_batch_id: actions[(2 * micro_batch_id + 1) * num_stage + num_stage - 1 - stage]

    # action placement
    for stage in range(num_stage):
        for mid in range(num_microbatch):
            f(stage, mid).device = torch.device.cuda(stage)
            b(stage, mid).device = torch.device.cuda(stage)

    # action in-device order
    stage_order = list()

    for stage in range(num_stage):
        order = list()
        num_warmup_microbatch = num_stage - stage - 1
        num_warmup_microbatch = min(num_warmup_microbatch, num_microbatch)
        num_microbatch_remain = num_microbatch - num_warmup_microbatch

        # warmup
        for mid in range(num_warmup_microbatch):
            order.append(f(stage, mid))
        
        # steady
        for i in range(num_microbatch_remain):
            f_mid = num_warmup_microbatch + i
            b_mid = i
            order.append(f(stage, f_mid))
            order.append(b(stage, b_mid))
        
        # cooldown
        for i in range(num_warmup_microbatch):
            b_mid = num_microbatch_remain + i
            order.append(b(stage, b_mid))
        
        stage_order.append(order)

    return stage_order
