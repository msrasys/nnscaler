import cube.algorithm.ops.memory as mem
from cube.graph.tensor import IRFullTensor, ValueMap


def test_transpose_dim_parallel():

    M = 512
    N = 1024
    input1 = IRFullTensor(shape=[M, N], name='input1').tosub()
    dim0 = 0
    dim1 = 1

    semantic_op = mem.Transpose(
        signature='torch.transpose', inputs=[input1, dim0, dim1], name='transpose'
    )
    semantic_op.infer_shape()
    print('semantic op:')
    print(semantic_op)

    op_dim = mem.TransposeDimParallel(semantic_op)
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
        assert isinstance(node, mem.Transpose)
    
    for idx, node in enumerate(nodes):
        print('=======')
        print(node)
        print('inputs:')
        for input in node.inputs():
            print(input)
            assert input.shape == [M // 4, N]
        print('outputs:')
        for output in node.outputs():
            print(output)
            assert output.shape == [N, M // 4]
            assert output.valmap == ValueMap(0, 1)
