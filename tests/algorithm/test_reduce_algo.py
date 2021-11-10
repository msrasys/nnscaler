from cube.graph.operator.function import Reduce
from cube.algorithm.reduce import ReduceDataParallel
from cube.graph.tensor import IRFullTensor, ValueMap


def test_elementwise_data_parallel():

    input1 = IRFullTensor(shape=[1024, 1024], name='input1').tosub()

    semantic_op = Reduce(
        signature='torch.sum', inputs=[input1], name='add'
    )
    semantic_op.infer_shape()
    print('semantic op:')
    print(semantic_op)

    op_dp = ReduceDataParallel(semantic_op)

    assert op_dp.chunk_num is None

    # test satisfy
    assert op_dp.satisfy(dict(chunk_num = 4))
    assert not op_dp.satisfy(dict(chunk_num = 10))
    
    nodes = op_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, Reduce)
    
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
