from typing import List

# spatial
def select(tensor, indices, val_op, shape): pass
def assign(tensor, ranks: List): pass

# temporal
def merge(su1, su2): pass


def spolicy(model, runtime_info, tp_size, dp_size, pp_size):

    n_devices = runtime_info.ndevs

    # each op is divided in (mp_dsize, dp_size)
    # and put in (pp_size) stage
    # TODO groups[stage][dp_group][tp_group] = devices (List[int])
    groups = parallel_group(n_devices, tp_size, dp_size, pp_size)

    # pipeline stage
    total_nodes = len(model.nodes())
    num_op_per_stage = total_nodes // pp_size
    for idx, op in enumerate(model.nodes()):
        pp_stage = idx // num_op_per_stage
        op.group = [pp_stage]

    # data parallel
    for op in model.nodes():
        # data parallel algorithm (suppose at index 0)
        dp_algo = op.logical_op.dist_algo(0)
        sub_graph = select(
            op = op, 
            algorithm = dp_algo, 
            config = dict(chunk_num=dp_size, uniform=True)
        )
        for dp_stage, dp_op in sub_graph.nodes():
            dp_op.group.append(dp_stage)
        model.replace(op, sub_graph)

    # tensor parallel
    # a transformer attention layer: 
    #   [attention: col_split(mm + mm + mm) + row_split(mm)]
    # a transformer feedforward layer:
    #   [feedforwrd: col_split(mm) + row_split(mm)]
    for idx in range(total_nodes):
        for dp_rank in range(dp_size):
            op = model.nodes(dp_size * idx + dp_rank)
            devices = op.devices
            sub_graph = None
            # Attention block
            # [1st linear -> 2nd linear)
            if first_to_2nd_linear(op):
                # split column
                tp_col_algo = op.logical_op.dist_algo(1)
                sub_graph = select(
                    op = op,
                    algorithm = tp_col_algo,
                    config = dict(chunk_num=tp_size, uniform=True)
                )
            # 2nd linear
            elif is_2nd_linear(op):
                # split row
                tp_row_algo = op.logical_op.dist_algo(2)
                sub_graph = select(
                    op = op,
                    algorithm = tp_row_algo,
                    config = dict(chunk_num=tp_size, uniform=True)
                )
            # MLP block
            # [3rd linear -> 4th linear]
            elif thrid_to_4th_linear(op):
                # split column
                tp_col_algo = op.logical_op.dist_algo(1)
                sub_graph = select(
                    op = op,
                    algorithm = tp_col_algo,
                    config = dict(chunk_num=tp_size, uniform=True)
                )
            elif is_4th_linear(op):
                # split row
                tp_row_algo = op.logical_op.dist_algo(2)
                sub_graph = select(
                    op = op,
                    algorithm = tp_row_algo,
                    config = dict(chunk_num=tp_size, uniform=True)
                )
            # else: no change, do redundant computation
            if sub_graph:
                for tp_stage, op in enumerate(sub_graph):
                    op.group.append(tp_stage)
                model.replace(op, sub_graph)
    # device assignment
    for op in model.nodes():
        pp_stage, dp_stage, tp_stage = op.group
        device = groups[pp_stage][dp_stage][tp_stage]
        assign(op, device)
    return model


def tpolicy(sus, relations, tp_size, pp_size, num_microbatch):
    """
    Pipeline 1f1b policy description -- generate a sequence

    Actions: a list of actions

    relations: list[(Action1, Action2)]: a list of tuples indicate partial order
    """

    # put sus to forward-backward sequences: List[List[SU(op)]]
    fb_op_seqs = list()
    for su in sus:
        for fb_seq in fb_op_seqs:
            if fb_seq[-1].happen_before(su):
                fb_seq.append(su)
                break
        else:
            fb_op_seqs.append([su])
    
    # merge to stages: List[List[SU(stage of ops)]]
    fb_stage_seqs = list()
    for fb_seq in fb_op_seqs:
        merged_su = fb_seq[0]
        merged_tag = fb_seq[0].tag
        for su in fb_seq[1]:
            if su.device == merged_su and su.tag == merged_tag:
                merged_su = merge(merged_su, su)
            else:
                fb_stage_seqs.append(merged_su)
                merged_su = su
                merged_tag = su.tag
        merged_su = merge(merged_su, su)

    # pp_size forward + pp_size backward
    assert (pp_size * 2 == len(fb_stage_seqs[0]))

    num_stage = pp_size

    f = lambda stage, micro_batch_id: fb_stage_seqs[micro_batch_id][stage]
    b = lambda stage, micro_batch_id: fb_stage_seqs[micro_batch_id][num_stage + stage]

    sequence = list()

    # warmup:
    for stage in range(num_stage):
        for mid in range(stage):
            sequence.append(f(stage, mid))
    
    # steady + cooldown:
    for mid in range(num_microbatch):
        # enqueue backward
        for stage in range(num_stage-1, -1, -1):
            sequence.append(b(stage, mid))
        # enqueue forward
        for stage in range(num_stage):
            f_mid = mid + 1 + num_stage - stage
            if f_mid >= num_microbatch:
                continue
            sequence.append(f(stage, f_mid))
    assert check_consistency(sequence, sus, relations)
    return sequence
