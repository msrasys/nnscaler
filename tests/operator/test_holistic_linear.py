"""
cmd for running the test

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=62000 \
    --use_env \
    tests/operator/test_holistic_linear.py
"""

from cube.tensor.logic.tensor import LogicalTensor

from cube.operator.holist.linear import LinearColumnWeight
from cube.operator.holist.linear import LinearColumnInputRowWeight

from cube.device.physic.group import DeviceGroup

import torch
import z3
torch.manual_seed(100)


def test_linear_POC():

    N = 1024
    input = torch.randn((1024, 1024)).cuda()
    weight = torch.randn((N, 1024))
    bias = torch.randn((N,))

    rank = DeviceGroup().rank

    # partial
    partial_weight = torch.chunk(weight, 4, dim=0)[rank].cuda()
    partial_bias = torch.chunk(bias, 4, dim=0)[rank].cuda()
    partial_out = torch._C._nn.linear(input, partial_weight, partial_bias)

    # full
    out_full = torch._C._nn.linear(input, weight.cuda(), bias.cuda())
    ref_out = torch.chunk(out_full, 4, dim=1)[rank].cuda()

    if rank == 0:
        print('max bias: ', torch.max(torch.abs(partial_out - ref_out)))
        print('sum bias: ', torch.sum(torch.abs(partial_out - ref_out)))


def test_holistic_linear_op_column_weight():
    """
    Note: Due to unknown reason in hardware, the output will have up to
    0.0001 bias. This is verified in `test_linear_POC()` The larger 
    K results larger bias.
    """
    N = 1024
    shapes = [(1024, 1024), (N, 1024), (N,), (1024, N)]
    input = LogicalTensor(shape=shapes[0])
    weight = LogicalTensor(shape=shapes[1])
    bias = LogicalTensor(shape=shapes[2])
    outputs = [LogicalTensor(shapes[3])]

    # ================================ Policy ===========================

    holistic_op = LinearColumnWeight(outputs, input, weight, bias)

    def policy(holist_op, input, weight, bias):
        solver = holist_op.solver
        attributes = holist_op.attributes
        input_layout = holist_op.input_layouts[0]
        weight_layout = holist_op.input_layouts[1]
        bias_layout = holist_op.input_layouts[2]
        output_layout = holist_op.output_layouts[0]

        # add restrictions based on device num
        device_num = torch.cuda.device_count()
        solver.add(weight_layout.chunk_num == 4)
        
        # iterate all configs
        configs = list()
        while solver.check() == z3.sat:
            config = solver.model()
            if DeviceGroup().rank == 0:
                print('find config: {}'.format(config))
            configs.append(config)
            solver.add(
                z3.Or([z3.Not(attr == config[attr]) for attr in attributes])
            )
            if len(attributes) == 0:
                break
        # choose one config -- suppose to the first
        config = configs[0]
        if DeviceGroup().rank == 0:
            print('selected config: {}'.format(config))

        # deploy decisions
        chunk_num = config[weight_layout.chunk_num].as_long()
        input_ranks = [list(range(0, chunk_num)),]
        weight_ranks = list()
        for rank in range(chunk_num):
            weight_ranks.append([rank])
        bias_ranks = weight_ranks

        return config, [input_ranks, weight_ranks, bias_ranks]
    
    # Missing Policy: where physical op executed?

    holistic_op.set_policy(policy)
    # ================================ Policy ===========================

    output = holistic_op(input, weight, bias)
    print('segments: {}'.format(len(output.segments)))

    # =============================== Test ==============================
    output_ref = torch._C._nn.linear(
        input.data.cuda(), weight.data.cuda(), bias.data.cuda()
    )
    rank = DeviceGroup().rank
    output_ref = torch.chunk(output_ref, chunks=4, dim=1)[rank].contiguous()
    out = output.get_physical_tensor(rank)
    # if rank == 0:
    #     print('ref: ', output_ref)
    #     print('get: ', out)
    #     print('max bias: ', torch.max(torch.abs(out - output_ref)))
    #     print('sum bias: ', torch.sum(torch.abs(out - output_ref)))
    error_max = torch.max(torch.abs(out - output_ref))
    assert error_max.item() < 2e-4
    # =============================== Test ==============================


def test_holistic_linear_op_column_input_row_weight():

    N = 1024
    shapes = [(1024, 1024), (N, 1024), (N,), (1024, N)]
    input = LogicalTensor(shape=shapes[0])
    weight = LogicalTensor(shape=shapes[1])
    bias = LogicalTensor(shape=shapes[2])
    outputs = [LogicalTensor(shapes[3])]

    # ================================ Policy ===========================

    holistic_op = LinearColumnInputRowWeight(outputs, input, weight, bias)

    def policy(holist_op, input, weight, bias):
        solver = holist_op.solver
        attributes = holist_op.attributes
        input_layout = holist_op.input_layouts[0]
        weight_layout = holist_op.input_layouts[1]
        bias_layout = holist_op.input_layouts[2]
        output_layout = holist_op.output_layouts[0]

        # add restrictions based on device num
        device_num = torch.cuda.device_count()
        solver.add(weight_layout.chunk_num == 4)
        
        # iterate all configs
        configs = list()
        while solver.check() == z3.sat:
            config = solver.model()
            if DeviceGroup().rank == 0:
                print('find config: {}'.format(config))
            configs.append(config)
            solver.add(
                z3.Or([z3.Not(attr == config[attr]) for attr in attributes])
            )
            if len(attributes) == 0:
                break
        # choose one config -- suppose to the first
        config = configs[0]
        if DeviceGroup().rank == 0:
            print('selected config: {}'.format(config))

        # deploy decisions
        chunk_num = config[weight_layout.chunk_num].as_long()
        input_ranks = list()
        for rank in range(chunk_num):
            input_ranks.append([rank])
        weight_ranks = input_ranks
        bias_ranks = weight_ranks

        return config, [input_ranks, weight_ranks, bias_ranks]

    # Missing Policy: where physical op executed?

    holistic_op.set_policy(policy)
    # ================================ Policy ===========================

    output = holistic_op(input, weight, bias)
    print('segments: {}'.format(len(output.segments)))

    # =============================== Test ==============================
    rank = DeviceGroup().rank
    input_ref = torch.chunk(input.data.cuda(), chunks=4, dim=-1)[rank]
    weight_ref = torch.chunk(weight.data.cuda(), chunks=4, dim=1)[rank]
    bias_ref = bias.data.cuda() / 4
    # if rank == 0:
    #     print('input ref: ', input_ref)
    #     print('weight ref: ', weight_ref)
    #     print('bias ref: ', bias_ref)

    output_ref = torch._C._nn.linear(
        input_ref, weight_ref, bias_ref
    )
    out = output.get_physical_tensor(rank)
    # if rank == 0:
    #     print('ref: ', output_ref)
    #     print('get: ', out)
    #     print('max bias: ', torch.max(torch.abs(out - output_ref)))
    #     print('sum bias: ', torch.sum(torch.abs(out - output_ref)))
    error_max = torch.max(torch.abs(out - output_ref))
    assert error_max.item() < 2e-4
    # =============================== Test ==============================


if __name__ == '__main__':
    group = DeviceGroup()
    
    # test_linear_POC()
    test_holistic_linear_op_column_weight()
    test_holistic_linear_op_column_input_row_weight()
