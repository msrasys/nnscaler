from cube.graph.operator.function import ElementWise
import cube.algorithm.ops.layernorm as ln
from cube.graph.tensor import IRFullTensor


def test_elementwise_dim_parallel():

    input1 = IRFullTensor(shape=[1024, 512, 256], name='input1').tosub()
    normalized_shape = [256,]
    weight = IRFullTensor(shape=[256], name='weight').tosub()
    bias = IRFullTensor(shape=[256], name='bias').tosub()
    eps = 1e-5

    semantic_op = ln.LayerNorm(
        signature='torch.nn.functional.layernorm',
        inputs=[input1, normalized_shape, weight, bias, eps],
        name='layernorm'
    )
    semantic_op.infer_shape()
    print('semantic op:')
    print(semantic_op)

    op_dim = ln.LayerNormDimParallel(semantic_op)

    assert op_dim.chunk_num is None

    # test satisfy
    assert op_dim.satisfy(dict(dim=0, chunk_num = 4))
    assert op_dim.satisfy(dict(dim=1, chunk_num = 8))
    assert not op_dim.satisfy(dict(dim=2, chunk_num = 8))
    
    nodes = op_dim.instantiate(semantic_op, dict(dim=1, chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, ln.LayerNorm)
    
    for node in nodes:
        print(node)
        print('inputs:')

        input = node.inputs(0)
        print(input)
        assert input.shape == [1024, 512 // 4, 256]

        weight = node.inputs(2)
        print(weight)
        assert weight.shape == [256,]

        bias = node.inputs(3)
        print(bias)
        assert bias.shape == [256,]

        print('outputs:')
        for output in node.outputs():
            print(output)
            assert output.shape == [1024, 512 // 4, 256]

    op_dim = ln.LayerNormDimParallel(semantic_op, dim=0)
    nodes = op_dim.instantiate(semantic_op, dict(chunk_num=4))

    for node in nodes:
        print(node)
        print('inputs:')

        input = node.inputs(0)
        print(input)
        assert input.shape == [1024 // 4, 512, 256]

        weight = node.inputs(2)
        print(weight)
        assert weight.shape == [256,]

        bias = node.inputs(3)
        print(bias)
        assert bias.shape == [256,]

        print('outputs:')
        for output in node.outputs():
            print(output)
            assert input.shape == [1024 // 4, 512, 256]
