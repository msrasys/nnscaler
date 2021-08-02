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
import cube.tensor.logic.segment as sg

from cube.operator.holist.linear import LinearColumnWeight
from cube.operator.holist.linear import LinearColumnInputRowWeight

from cube.device.physic.group import DeviceGroup

import torch
torch.manual_seed(100)


class LogicalLinear:

    def __init__(self): pass

    def shape_infer(self, input_shape, weight_shape, bias_shape=None):
        """
        Return the outputs shape [list[int],]
        """
        #TODO: change all shape impl to list
        output_shape = list(input_shape.shape)
        output_shape[-1] = weight_shape.shape[0]
        return [output_shape]


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
    input = LogicalTensor(shape=(1024,1024))
    weight = LogicalTensor(shape=(N,1024))
    bias = LogicalTensor(shape=(N,))

    # output = LogicalLinear(input, weight, bias)

    # ================================ Policy ===========================

    holistic_op = LinearColumnWeight()
    holistic_op.logical_op = LogicalLinear()

    def policy_for_how_many_tiles(outliner):
        if isinstance(outliner, sg.outline.Full):
            pass
        elif isinstance(outliner, sg.outline.SplitAxis):
            if outliner.chunk_num.get() is None:
                outliner.chunk_num = 4
        else:
            raise TypeError("Unhandled outliner type")
    # -> together

    def policy_for_each_tile_placement(community, input, weight, bias):
        # generate results (hard code) [helper function]
        input_ranks = [
            [[0,1,2,3]], [DeviceGroup().all_ranks()]
            [[0],[1],[2],[3]],
            [[0],[1],[2],[3]]
        ]
        input_val_map_fns = list([None, None, None])
        return input_ranks, input_val_map_fns
    
    # Missing Policy: where physical op executed?

    holistic_op.set_deploy_policy(
        policy_for_each_tile_placement
    )
    holistic_op.set_segmentation_policy(
        policy_for_how_many_tiles
    )
    # ================================ Policy ===========================

    output = holistic_op(input, weight, bias)

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


if __name__ == '__main__':
    group = DeviceGroup()
    
    # test_linear_POC()
    test_holistic_linear_op_column_weight()