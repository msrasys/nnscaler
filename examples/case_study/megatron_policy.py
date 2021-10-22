from typing import List

from cube.schedule.su import SUType


def transform_policy(graph, resource):

    # suppose this is the policy config that both
    # transformation and schedule policy know
    tp_size = 8,
    pp_size = 4,
    dp_size = resource.ndev // (tp_size * pp_size)
    num_micro_batch = 16

    # each op is divided in (mp_dsize, dp_size)
    # and put in (pp_size) stage
    # TODO groups[stage][dp_group][tp_group] = devices (List[int])

    # data + pipeline parallelism: first transform graph
    for idx, op in enumerate(graph.nodes()):
        algorithm = op.algorithm('data_parallel')
        graph.partition(
            op, algorithm, config=dict(chunk_size=num_micro_batch * dp_size)
        )
        pp_stage = idx // (len(graph.nodes()) // pp_size)
        op.tag('pp_stage', pp_stage)

    # data parallel
    for op in graph.nodes():
        algorithm = op.algorithm('data_parallel')
        graph.partition(op)

    # tensor parallel
    # a transformer attention layer: 
    #   [attention: col_split(mm + mm + mm) + row_split(mm)]
    # a transformer feedforward layer:
    #   [feedforwrd: col_split(mm) + row_split(mm)]
    for idx, op in enumerate(graph.nodes()):
        # Attention block
        # [1st linear -> 2nd linear)
        if op_from_1st_to_2nd_linear(op):
            # split column
            tp_col_algo = op.logical_op.dist_algo(1)
            graph.partition(
                op = op,
                algorithm = tp_col_algo,
                config = dict(chunk_num=tp_size, uniform=True)
            )
        # 2nd linear
        elif op_is_2nd_linear(op):
            # split row
            tp_row_algo = op.logical_op.dist_algo(2)
            graph.partition(
                op = op,
                algorithm = tp_row_algo,
                config = dict(chunk_num=tp_size, uniform=True)
            )
        # MLP block
        # [3rd linear -> 4th linear]
        elif op_from_3rd_to_4th_linear(op):
            # split column
            tp_col_algo = op.logical_op.dist_algo(1)
            graph.partition(
                op = op,
                algorithm = tp_col_algo,
                config = dict(chunk_num=tp_size, uniform=True)
            )
        elif op_is_4th_linear(op):
            # split row
            tp_row_algo = op.logical_op.dist_algo(2)
            graph.partition(
                op = op,
                algorithm = tp_row_algo,
                config = dict(chunk_num=tp_size, uniform=True)
            )
    return graph


def schedule_policy(su_graph, resource):

    # suppose this is the policy config that both
    # transformation and schedule policy know
    tp_size = 8,
    pp_size = 4,
    dp_size = resource.ndev // (tp_size * pp_size)
    num_micro_batch = 16

    # given tp, pp, dp, num mirco batch, set the device id
    # for hierachical: [pipeline][data][tensor] = device (int)
    dev_groups = set_device_id(tp_size, dp_size, pp_size, num_micro_batch)

    # put sus to forward-backward sequences: List[List[SU(op)]]
    fb_op_sus = list()
    for su in su_graph.sus():
        if su.stype == SUType.Forward or su.stype == SUType.Backward:
            for fb_seq in fb_op_sus:
                if fb_seq[-1].happen_before(su):
                    fb_seq.append(su)
                    break
            else:
                fb_op_sus.append([su])
    
    # merge to stages: List[List[SU(stage sequential of ops)]]
    fb_stage_sus = list()
    assert len(fb_op_sus) == tp_size * dp_size * num_micro_batch
    for dp in range(dp_size):
        for tp in range(tp_size):
            fb_stage_sus.append([])
            fb_sus = fb_op_sus[dp * dp_size + tp]
            for idx, su in enumerate(fb_sus):
                pp = idx // ( len(fb_sus) // pp_size)
                device = dev_groups[pp][dp][tp]
                su_graph.assign(su, device)
            merged_su = None
            for su in fb_sus:
                if merged_su is None:
                    merged_su = su
                    fb_stage_sus[-1].append([su])
                else:
                    # same device op can be merged
                    merged_su = su_graph.merge(merged_su, su)

    num_stage = pp_size
    f = lambda stage, micro_batch_id: fb_stage_sus[micro_batch_id][stage]
    b = lambda stage, micro_batch_id: fb_stage_sus[micro_batch_id][num_stage + stage]

    sequence = list()

    # warmup:
    for stage in range(num_stage):
        for mid in range(stage):
            sequence.append(f(stage, mid))
    
    # steady + cooldown:
    for mid in range(num_micro_batch):
        # enqueue backward
        for stage in range(num_stage-1, -1, -1):
            sequence.append(b(stage, mid))
        # enqueue forward
        for stage in range(num_stage):
            f_mid = mid + 1 + num_stage - stage
            if f_mid >= num_micro_batch:
                continue
            sequence.append(f(stage, f_mid))

    # infor system the control dependency by topological assignment
    su_graph.set_order(sequence)
    return su_graph
