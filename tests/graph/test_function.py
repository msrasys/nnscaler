from cube.graph.operator.function import Linear
from cube.graph.tensor import IRFullTensor
from cube.algorithm.linear import LinearDataParallel


def test_linear_algo():

    input = IRFullTensor(shape=[1024, 1024], name='input').tosub()
    weight = IRFullTensor(shape=[1000, 1024], name='weight').tosub()
    bias = IRFullTensor(shape=[1000,], name='bias').tosub()

    semantic_op = Linear(
        signature='torch.nn.functional.linear',
        inputs = [input, weight, bias],
    )
    semantic_op.infer_shape()

    assert len(semantic_op.algorithms()) == 3
    assert isinstance(semantic_op.algorithms('data'), LinearDataParallel)
