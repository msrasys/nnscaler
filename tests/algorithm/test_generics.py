from cube.algorithm.generics import GenericDistAlgo
from cube.ir.cten import IRCell, IRTensor


def test_generic_algo_init():
    input1 = IRTensor(shape=[1024, 1024])
    input2 = IRTensor(shape=[1024, 1000])
    bias = None
    cell = IRCell(name='test', signature='test', input_length=3, output_length=1)
    cell.set_input(0, input1)
    cell.set_input(1, input2)
    cell.set_input(2, bias)
    cell.outputs(0).shape = [1024, 1000]

    algo = GenericDistAlgo(cell)
    assert algo.logic_op is IRCell
    assert len(algo.input_shapes) == 3
    assert algo.input_shapes[0] == [1024, 1024]
    assert algo.input_shapes[1] == [1024, 1000]
    assert algo.input_shapes[2] is None
    assert len(algo.output_shapes) == 1
    assert algo.output_shapes[0] == [1024, 1000]
