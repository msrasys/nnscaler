import cube.algorithm.ops.activation as activation
from cube.graph.operator.function import Dropout
from cube.graph.tensor import IRFullTensor


def test_softmax_dim_parallel():

    input1 = IRFullTensor(shape=[1024, 1024], name='input1').tosub()
    dim = -1
    stacklevel = 3
    dtype = None

    semantic_op = activation.Softmax(
        signature = 'torch.nn.functional.softmax',
        inputs = [input1, dim, stacklevel, dtype],
    )
    semantic_op.infer_shape()

    op_dim = activation.SoftmaxDimParallel(semantic_op)
    assert op_dim.dim is None
    assert op_dim.chunk_num is None

    assert op_dim.satisfy(dict(dim=0, chunk_num=4))
    assert op_dim.satisfy(dict(dim=-2, chunk_num=4))
    assert not op_dim.satisfy(dict(dim=1, chunk_num=4))
    assert not op_dim.satisfy(dict(dim=-1, chunk_num=4))

    nodes = op_dim.instantiate(semantic_op, dict(dim=0, chunk_num=4))
    for node in nodes:
        print(node)
        assert isinstance(node, activation.Softmax)
        for input in node.inputs():
            print(input)
            assert input.shape == [1024 // 4, 1024]
        for output in node.outputs():
            print(output)
            assert output.shape == [1024 // 4, 1024]
        assert node.kwargs == semantic_op.kwargs
        assert node.stay_dims == semantic_op.stay_dims


def test_dropout_dim_parallel():

    input1 = IRFullTensor(shape=[1024, 1024], name='input1').tosub()
    p = 0.5
    training = True
    inplace = False

    semantic_op = activation.Dropout(
        signature = 'torch.nn.functional.softmax',
        inputs = [input1, p, training, inplace],
    )
    semantic_op.infer_shape()

    op_dim = activation.DropoutDimParallel(semantic_op)
    assert op_dim.dim is None
    assert op_dim.chunk_num is None

    assert op_dim.satisfy(dict(dim=0, chunk_num=4))
    assert op_dim.satisfy(dict(dim=-2, chunk_num=4))
    assert op_dim.satisfy(dict(dim=1, chunk_num=4))
    assert op_dim.satisfy(dict(dim=-1, chunk_num=4))

    nodes = op_dim.instantiate(semantic_op, dict(dim=0, chunk_num=4))
    for node in nodes:
        print(node)
        assert isinstance(node, activation.Dropout)
        for input in node.inputs():
            print(input)
            assert input.shape == [1024 // 4, 1024]
        for output in node.outputs():
            print(output)
            assert output.shape == [1024 // 4, 1024]
        assert node.kwargs == semantic_op.kwargs
        assert node.stay_dims == semantic_op.stay_dims
