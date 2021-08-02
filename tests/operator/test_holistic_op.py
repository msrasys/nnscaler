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

import cube.tensor.logic.outline as outline
from cube.tensor.logic.tensor import LogicalTensor
from cube.operator.holist.generics import GenericHolisticOp
from cube.device.physic.group import DeviceGroup
import torch
import z3

def test_generic_holistic_op_init():

    shapes = [(32, 2048), (1024, 2048), (32, 1024)]
    op = GenericHolisticOp(shapes)

    # description
    input_layout = outline.Full(
        op.solver, op.shapes[0],
    )
    weight_layout = outline.SplitAxis(
        op.solver, op.shapes[1],
        axis=0, chunk_num=None, overlap=0,
    )
    output_layout = outline.SplitAxis(
        op.solver, op.shapes[2],
        axis=0, chunk_num=weight_layout.chunk_num, overlap=0,
    )

    assert op.shapes == shapes
    assert len(op.input_layouts) == 0
    assert len(op.output_layouts) == 0
    assert op.logical_op is None
    assert op.policy_fn is None

    op.set_input_layouts([input_layout, weight_layout])
    op.set_output_layouts([output_layout])

    assert len(op.input_layouts) == 2
    assert len(op.output_layouts) == 1
    assert len(op.attributes) == 5


def test_generic_holistic_op_input_adapter():

    shapes = [(32, 512), (1024, 512), (32, 1024)]
    input = LogicalTensor(shape=shapes[0])
    weight = LogicalTensor(shape=shapes[1])

    op = GenericHolisticOp(shapes)

    # description
    input_layout = outline.Full(
        op.solver, op.shapes[0],
    )
    weight_layout = outline.SplitAxis(
        op.solver, op.shapes[1],
        axis=0, chunk_num=None, overlap=0,
    )
    output_layout = outline.SplitAxis(
        op.solver, op.shapes[2],
        axis=0, chunk_num=weight_layout.chunk_num, overlap=0,
    )

    op.set_input_layouts([input_layout, weight_layout])
    op.set_output_layouts([output_layout])

    def policy(holist_op):
        solver = holist_op.solver
        attributes = holist_op.attributes
        input_layout = holist_op.input_layouts[0]
        weight_layout = holist_op.input_layouts[1]
        output_layout = holist_op.output_layouts[0]

        # add restrictions based on device num
        device_num = torch.cuda.device_count()
        solver.add(weight_layout.chunk_num <= 4)
        
        # iterate all configs
        configs = list()
        while solver.check() == z3.sat:
            config = solver.model()
            configs.append(config)
            solver.add(
                z3.Or([z3.Not(attr == config[attr]) for attr in attributes])
            )
            if len(attributes) == 0:
                break
        # choose one config -- suppose to the first
        config = configs[0]

        # deploy decisions
        chunk_num = config[weight_layout.chunk_num].as_long()
        input_ranks = [list(range(0, chunk_num)),]
        weight_ranks = list()
        for rank in range(chunk_num):
            weight_ranks.append([rank])

        return config, [input_ranks, weight_ranks]

    op.set_policy(policy)
    op.input_adapter(input, weight)


if __name__ == '__main__':
    group = DeviceGroup()
    test_generic_holistic_op_init()
    test_generic_holistic_op_input_adapter()
