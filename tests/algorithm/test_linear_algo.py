from cube.graph.operator.function import Linear
from cube.algorithm.ops.linear import LinearDataParallel
from cube.algorithm.ops.linear import LinearColumnWeight
from cube.algorithm.ops.linear import LinearRowWeight
from cube.graph.tensor import IRFullTensor, ValueMap


def test_linear_data_parallel():

    input = IRFullTensor(shape=[1024, 1024], name='input').tosub()
    weight = IRFullTensor(shape=[1000, 1024], name='weight').tosub()
    bias = IRFullTensor(shape=[1000,], name='bias').tosub()

    semantic_op = Linear(
        signature='torch.nn.functional.linear',
        inputs = [input, weight, bias],
    )
    semantic_op.infer_shape()

    linear_dp = LinearDataParallel(semantic_op)

    assert linear_dp.chunk_num is None

    # test satisfy
    assert linear_dp.satisfy(dict(chunk_num=4))
    assert not linear_dp.satisfy(dict(chunk_num=10))

    nodes = linear_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, Linear)

    inputs = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in inputs:
        print(input)
    weights = [node.inputs(1) for node in nodes]
    print('weights:')
    for weight in weights:
        print(weight)
    biass = [node.inputs(2) for node in nodes]
    print('bias:')
    for bias in biass:
        print(bias)
    
    for idx, x in enumerate(inputs):
        assert x.shape == [256, 1024]
        assert x.indices.get()[0] == slice(256 * idx, 256 * (idx + 1), 1)
    assert not inputs[0].overlap(inputs[1])
    assert not inputs[0].overlap(inputs[2])
    assert not inputs[0].overlap(inputs[3])

    for w in weights:
        assert w.shape == [1000, 1024]
        assert w == weight

    for b in biass:
        assert b.shape == [1000]
        assert b == bias


def test_linear_column_weight():
    input = IRFullTensor(shape=[1024, 1024], name='input').tosub()
    weight = IRFullTensor(shape=[1000, 1024], name='weight').tosub()
    bias = IRFullTensor(shape=[1000,], name='bias').tosub()

    semantic_op = Linear(
        signature='torch.nn.functional.linear',
        inputs = [input, weight, bias],
    )
    semantic_op.infer_shape()

    linear_col_weight = LinearColumnWeight(semantic_op)

    # test satisfy
    assert linear_col_weight.satisfy(dict(chunk_num=4))
    assert linear_col_weight.satisfy(dict(chunk_num=10))
    assert not linear_col_weight.satisfy(dict(chunk_num=12))

    nodes = linear_col_weight.instantiate(semantic_op, config=dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, Linear)

    inputs = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in inputs:
        print(input)
    weights = [node.inputs(1) for node in nodes]
    print('weights:')
    for weight in weights:
        print(weight)
    biass = [node.inputs(2) for node in nodes]
    print('bias:')
    for bias in biass:
        print(bias)
    outputs = [node.outputs(0) for node in nodes]
    print('output:')
    for output in outputs:
        print(output)

    for x in inputs:
        assert x == input

    for idx, w in enumerate(weights):
        assert w.shape == [250, 1024]
        assert w.indices.get()[0] == slice(250 * idx, 250 * (idx + 1), 1)

    for idx, b in enumerate(biass):
        assert b.shape == [250]
        assert b.indices.get() == (slice(250 * idx, 250 * (idx + 1), 1),)
    
    for idx, output in enumerate(outputs):
        assert output.shape == [1024, 250]
        assert output.indices.get()[0] == slice(0, 1024, 1)
        assert output.indices.get()[1] == slice(250 * idx, 250 * (idx + 1), 1)


def test_linear_row():
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

    linear_row_weight = LinearRowWeight(semantic_op)

    # test satisfy
    assert linear_row_weight.satisfy(dict(chunk_num=4))
    assert not linear_row_weight.satisfy(dict(chunk_num=10))
    assert not linear_row_weight.satisfy(dict(chunk_num=12))

    nodes = linear_row_weight.instantiate(semantic_op, config=dict(chunk_num=4))

    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, Linear)

    inputs = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in inputs:
        print(input)
    weights = [node.inputs(1) for node in nodes]
    print('weights:')
    for weight in weights:
        print(weight)
    biass = [node.inputs(2) for node in nodes]
    print('bias:')
    for bias in biass:
        print(bias)
    outputs = [node.outputs(0) for node in nodes]
    print('output:')
    for output in outputs:
        print(output)

    for idx, x in enumerate(inputs):
        assert x.shape == [1024, 256]
        assert x.indices.get()[1] == slice(256 * idx, 256 * (idx + 1), 1)
        assert x.val_map == ValueMap(0, 1)

    for idx, w in enumerate(weights):
        assert w.shape == [1000, 256]
        assert w.indices.get()[1] == slice(256 * idx, 256 * (idx + 1), 1)
        assert w.val_map == ValueMap(0, 1)

    for idx, b in enumerate(biass):
        assert b.shape == [1000,]
        assert b.indices.get()[0] == slice(0, 1000, 1)
        assert b.val_map == ValueMap(idx, 4)

    for idx, output in enumerate(outputs):
        assert output.shape == [1024, 1000]
        assert output.val_map == ValueMap(idx, 4)
