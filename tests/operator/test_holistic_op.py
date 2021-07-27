"""
cmd for running the test

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=62000 \
    --use_env \
    tests/operator/test_holistic_op.py
"""

import cube.tensor.logic.segment as sg
from cube.tensor.logic.tensor import LogicalTensor
from cube.operator.holist.generics import GenericHolisticOp
from cube.device.physic.group import DeviceGroup
import torch

def test_generic_holistic_op_init():

    # description
    input_layout = sg.SplitAxis(
        axis=0, overlap=0, reduction=sg.ReductionOp.Replica
    )
    weight_layout = sg.Full(reduction=sg.ReductionOp.Replica)
    output_layout = sg.SplitAxis(
        axis=0, overlap=0, chunk_num=input_layout.chunk_num,
        reduction=sg.ReductionOp.Replica
    )

    op = GenericHolisticOp(
        input_layout=[input_layout, weight_layout],
        output_layout=[output_layout],
        input_format=[None, None],
        output_format=[None],
    )

    assert len(op.input_layout) == 2
    assert len(op.input_format) == 2
    assert len(op.output_layout) == 1
    assert len(op.output_format) == 1
    assert op.logical_op is None
    assert op.policy_fn is None


def test_generic_holistic_op_input_adapter():

    input_layout = sg.SplitAxis(
        axis=0, overlap=0, reduction=sg.ReductionOp.Replica
    )
    weight_layout = sg.Full(reduction=sg.ReductionOp.Replica)
    output_layout = sg.SplitAxis(
        axis=0, overlap=0, chunk_num=input_layout.chunk_num,
        reduction=sg.ReductionOp.Replica
    )

    op = GenericHolisticOp(
        input_layout=[input_layout, weight_layout],
        output_layout=[output_layout],
        input_format=[None, None],
        output_format=[None],
    )

    input = LogicalTensor(shape=(1024, 1024))
    weight = LogicalTensor(shape=(1024, 1024))

    ## Policy Here
    input_layout.chunk_num = 4
    assert output_layout.chunk_num.get() == 4
    def policy_fn(input_communities, input, weight):
        input_ranks = [
            [[0],[1],[2],[3]],
            [[0,1,2,3]]
        ]
        input_val_map_fns = list([None, None])
        return input_ranks, input_val_map_fns

    op.register_policy(policy_fn)
    op.input_adapter(input, weight)

    myrank = DeviceGroup().rank
    assert len(input.communities) == 4
    assert len(weight.communities) == 1
    physical_tensor = input.get_physical_tensor(input.segments[myrank])
    piece = 1024 // 4
    start = int(myrank * piece)
    assert torch.allclose(physical_tensor, input.data.cuda()[start:start+piece, :]) is True
    physical_tensor = weight.get_physical_tensor(weight.segments[0])
    assert torch.allclose(physical_tensor, weight.data.cuda()) is True


if __name__ == '__main__':
    group = DeviceGroup()
    test_generic_holistic_op_init()
    test_generic_holistic_op_input_adapter()
