import cube.algorithm.ops.reduce as reduce
from cube.graph.tensor import IRFullTensor, ValueMap


def test_reduce_dim_parallel():

    input1 = IRFullTensor(shape=[1024, 1024], name='input1').tosub()
    dim = None

    semantic_op = reduce.Sum(
        signature='torch.sum', inputs=[input1, dim], name='add'
    )
    semantic_op.infer_shape()
    print('semantic op:')
    print(semantic_op)

    op_dim = reduce.SumDimParallel(semantic_op)
    assert op_dim.dim is None
    assert op_dim.chunk_num is None

    # test satisfy
    assert op_dim.satisfy(dict(dim=0, chunk_num=4))
    assert op_dim.satisfy(dict(dim=-2, chunk_num=4))
    assert op_dim.satisfy(dict(dim=1, chunk_num=4))
    assert op_dim.satisfy(dict(dim=-1, chunk_num=4))
    
    nodes = op_dim.instantiate(semantic_op, dict(dim=0, chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, reduce.Sum)
    
    for idx, node in enumerate(nodes):
        print('=======')
        print(node)
        print('inputs:')
        for input in node.inputs():
            print(input)
            assert input.shape == [256, 1024]
        print('outputs:')
        for output in node.outputs():
            print(output)
            assert output.shape == [1]
            assert output.val_map == ValueMap(idx, 4)


    dim = 1
    semantic_op = reduce.Sum(
        signature='torch.sum', inputs=[input1, dim], name='add'
    )
    semantic_op.infer_shape()
    assert op_dim.satisfy(dict(dim=0, chunk_num=4))
    assert op_dim.satisfy(dict(dim=-2, chunk_num=4))
    assert op_dim.satisfy(dict(dim=1, chunk_num=4))
    assert op_dim.satisfy(dict(dim=-1, chunk_num=4))

    op_dim = reduce.SumDimParallel(semantic_op)
    nodes = op_dim.instantiate(semantic_op, dict(dim=0, chunk_num=4))
    for idx, node in enumerate(nodes):
        print(node)
        print('inputs:')
        for input in node.inputs():
            print(input)
            assert input.shape == [256, 1024]
        print('outputs:')
        for output in node.outputs():
            print(output)
            assert output.shape == [256]
            assert output.val_map == ValueMap(0, 1)
