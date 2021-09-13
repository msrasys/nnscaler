import torch
 
def select(tensor, indices, val_map_op=None, shape=None): pass

def input_adapter(inputs, target): pass

def iter_op(DAG): pass
def generate_for_each_rank(pDAG): pass


def sschedule_dp(pDAG, resources, input_tensors):
    """
    Data Parallel Description

    Args:
        * pDAG: (partial) logical computation graph
        * Resources: Environment inlcuding devices, network topology etc
    Returns:
        * pDAGs (list[DAG]) execution (local & physical) DAG for each rank
    """
    # rank [0,1,..., pp_size-1], [pp_size, ..., 2*pp_size - 1], ...
    ndevs = resources.ndevs
    # suppose 8 devices, 4 for pipeline, 2 for data parallel
    dp_size = 2
    pp_size = 4
    for op in iter_op(pDAG):
        for op_id, dist_op in enumerate(op.dist_candidates()):
            # find the data parallelism
            if is_data_parallelism(dist_op):
                for tensor in dist_op.inputs + dist_op.outputs:
                    if isinstance(tensor.segment, SplitAxis):
                        # pipeline micro-batch = 4
                        tensor.segment.chunk_num = dp_size * 4
                        # translate to logical tensor segments
                        tensor.segment.translate()
                dist_op.generate_ops()
                # setup placement
                stage = op_id // (len(pDAG) // pp_size)
                for dp_id, sub_op in enumerate(dist_op.ops):
                    sub_op.device = (dp_id % dp_size) * pp_size + stage
                # materialize -- call to the deploy
                dist_op.materialize()
                # generate input adapter
                pDAG.replace(op, dist_op)
                break
    return pDAG


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
