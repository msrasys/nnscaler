from cube.graph.operator.function import Linear
from cube.algorithm.linear import LinearDataParallel, LinearColumnWeight, LinearRowWeight
from cube.graph.tensor import IRFullTensor


def test_linear_data_parallel():

    input = IRFullTensor(shape=[1024, 1024], name='input').tosub()
    weight = IRFullTensor(shape=[1000, 1024], name='weight').tosub()
    bias = IRFullTensor(shape=[1000,], name='bias').tosub()

    semantic_op = Linear(
        signature='torch.nn.functional.linear',
        inputs = [input, weight, bias],
    )
    semantic_op.infer_shape()

    input_shapes = list()
    for input in semantic_op.inputs():
        input_shapes.append(input.shape)

    output_shapes = list()
    for output in semantic_op.outputs():
        output_shapes.append(output.shape)

    linear_dp = LinearDataParallel(
        input_shapes=input_shapes,
        output_shapes=output_shapes,
    )

    assert linear_dp.chunk_num is None

    # test satisfy
    assert linear_dp.satisfy(dict(chunk_num=4))
    assert not linear_dp.satisfy(dict(chunk_num=10))

    nodes = linear_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, Linear)

    inputs = [node.inputs(0) for node in nodes]
    weights = [node.inputs(1) for node in nodes]
    biass = [node.inputs(2) for node in nodes]
    
    for x in inputs:
        assert x.shape == [256, 1024]
    assert not inputs[0].overlap(inputs[1])
    assert not inputs[0].overlap(inputs[2])
    assert not inputs[0].overlap(inputs[3])

    for w in weights:
        assert w.shape == [1000, 1024]
        assert w == weight

    for b in biass:
        assert b.shape == [1000]
        assert b == bias
