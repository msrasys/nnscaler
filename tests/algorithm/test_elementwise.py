from cube.graph.operator.function import ElementWise
from cube.algorithm.ops.elementwise import ElementWiseDimParallel
from cube.graph.tensor import IRFullTensor


def test_elementwise_dim_parallel():

    input1 = IRFullTensor(shape=[1024, 1024], name='input1').tosub()
    input2 = IRFullTensor(shape=[1024, 1024], name='input2').tosub()

    semantic_op = ElementWise(
        signature='torch.add', inputs=[input1, input2], name='add'
    )
    semantic_op.infer_shape()
    print('semantic op:')
    print(semantic_op)

    op_dp = ElementWiseDimParallel(semantic_op, dim=0)

    assert op_dp.chunk_num is None

    # test satisfy
    assert op_dp.satisfy(dict(chunk_num = 4))
    assert not op_dp.satisfy(dict(chunk_num = 10))
    
    nodes = op_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, ElementWise)
    
    for node in nodes:
        print('=======')
        print(node)
        print('inputs:')
        for input in node.inputs():
            print(input)
            assert input.shape == [256, 1024]
        print('outputs:')
        for output in node.outputs():
            print(output)
            assert output.shape == [256, 1024]

    op_dp = ElementWiseDimParallel(semantic_op, dim=1)
    nodes = op_dp.instantiate(semantic_op, dict(chunk_num=4))

    for node in nodes:
        print('=======')
        print(node)
        print('inputs:')
        for input in node.inputs():
            print(input)
            assert input.shape == [1024, 256]
        print('outputs:')
        for output in node.outputs():
            print(output)
            assert output.shape == [1024, 256]
