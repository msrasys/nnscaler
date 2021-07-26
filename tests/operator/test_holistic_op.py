import cube.tensor.logic.segment as sg
from cube.tensor.logic.tensor import LogicalTensor

from cube.operator.holist.generics import GenericHolisticOp


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
    


if __name__ == '__main__':

    test_generic_holistic_op_init()