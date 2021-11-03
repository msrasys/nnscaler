from cube.algorithm.generics import GenericDistAlgo
from cube.graph.operator.function import Linear


def test_generic_algo_init():

    algo = GenericDistAlgo(
        input_shapes=[[1024,1024], [1024, 1024], None],
        output_shapes=[[1024, 1024]]
    )
    assert algo.logical_op is None


def test_generic_set_logic_op():

    algo = GenericDistAlgo(
        input_shapes=[[1024,1024], [1024, 1024], None],
        output_shapes=[[1024, 1024]]
    )
    algo.set_logic_op(Linear)
    assert algo.logical_op == Linear
