import cube.algorithm.ops.bmm as bmm
from cube.graph.tensor import IRFullTensor, ValueMap


def test_bmm_data_parallel():

    B = 64      # seq len
    N = 256      # batch
    M = 1024    # hiddend size = dim_head * num_head
    P = 512
    input1 = IRFullTensor(shape=[B, N, M], name='hidden').tosub()
    input2 = IRFullTensor(shape=[B, M, P], name='input2').tosub()

    semantic_op = bmm.BatchLinear(
        signature='torch.bmm',
        inputs = [input1, input2]
    )
    semantic_op.infer_shape()

    bmm_dp = bmm.BatchLinearDataParallel(semantic_op)

    assert bmm_dp.chunk_num is None

    assert bmm_dp.satisfy(dict(chunk_num=8))
    assert not bmm_dp.satisfy(dict(chunk_num=9))

    nodes = bmm_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, bmm.BatchLinear)

    nodes = bmm_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, bmm.BatchLinear)
    
    input1s = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in input1s:
        print(input)
        assert input.shape == [B // 4, N, M]

    input2s = [node.inputs(1) for node in nodes]
    print('input2s:')
    for input2 in input2s:
        print(input2)
        assert input2.shape == [B // 4, M, P]

    outputs = [node.outputs(0) for node in nodes]
    for output in outputs:
        print(output)
        assert output.shape == [B // 4, N, P]
        assert output.valmap == ValueMap(0, 1)


def test_bmm_n_parallel():

    B = 64      # seq len
    N = 256      # batch
    M = 1024    # hiddend size = dim_head * num_head
    P = 512
    input1 = IRFullTensor(shape=[B, N, M], name='hidden').tosub()
    input2 = IRFullTensor(shape=[B, M, P], name='input2').tosub()

    semantic_op = bmm.BatchLinear(
        signature='torch.bmm',
        inputs = [input1, input2]
    )
    semantic_op.infer_shape()

    bmm_dp = bmm.BatchLinearNParallel(semantic_op)

    assert bmm_dp.chunk_num is None

    assert bmm_dp.satisfy(dict(chunk_num=8))
    assert not bmm_dp.satisfy(dict(chunk_num=9))

    nodes = bmm_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, bmm.BatchLinear)

    nodes = bmm_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, bmm.BatchLinear)
    
    input1s = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in input1s:
        print(input)
        assert input.shape == [B, N // 4, M]

    input2s = [node.inputs(1) for node in nodes]
    print('input2s:')
    for input2 in input2s:
        print(input2)
        assert input2.shape == [B, M, P]

    outputs = [node.outputs(0) for node in nodes]
    for output in outputs:
        print(output)
        assert output.shape == [B, N // 4, P]
        assert output.valmap == ValueMap(0, 1)


def test_bmm_m_parallel():

    B = 64
    N = 256
    M = 1024
    P = 512
    input1 = IRFullTensor(shape=[B, N, M], name='input1').tosub()
    input2 = IRFullTensor(shape=[B, M, P], name='input2').tosub()

    semantic_op = bmm.BatchLinear(
        signature='torch.bmm',
        inputs = [input1, input2]
    )
    semantic_op.infer_shape()

    bmm_dp = bmm.BatchLinearMParallel(semantic_op)

    assert bmm_dp.chunk_num is None

    assert bmm_dp.satisfy(dict(chunk_num=8))
    assert not bmm_dp.satisfy(dict(chunk_num=9))

    nodes = bmm_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, bmm.BatchLinear)

    nodes = bmm_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, bmm.BatchLinear)
    
    input1s = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in input1s:
        print(input)
        assert input.shape == [B, N, M // 4]

    input2s = [node.inputs(1) for node in nodes]
    print('input2s:')
    for input2 in input2s:
        print(input2)
        assert input2.shape == [B, M // 4, P]

    outputs = [node.outputs(0) for node in nodes]
    for idx, output in enumerate(outputs):
        print(output)
        assert output.shape == [B, N, P]
        assert output.valmap == ValueMap(idx, 4)


def test_bmm_p_parallel():

    B = 64      # seq len
    N = 256      # batch
    M = 1024    # hiddend size = dim_head * num_head
    P = 512
    input1 = IRFullTensor(shape=[B, N, M], name='hidden').tosub()
    input2 = IRFullTensor(shape=[B, M, P], name='input2').tosub()

    semantic_op = bmm.BatchLinear(
        signature='torch.bmm',
        inputs = [input1, input2]
    )
    semantic_op.infer_shape()

    bmm_dp = bmm.BatchLinearPParallel(semantic_op)

    assert bmm_dp.chunk_num is None

    assert bmm_dp.satisfy(dict(chunk_num=8))
    assert not bmm_dp.satisfy(dict(chunk_num=9))

    nodes = bmm_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, bmm.BatchLinear)

    nodes = bmm_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, bmm.BatchLinear)
    
    input1s = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in input1s:
        print(input)
        assert input.shape == [B, N, M]

    input2s = [node.inputs(1) for node in nodes]
    print('input2s:')
    for input2 in input2s:
        print(input2)
        assert input2.shape == [B, M, P // 4]

    outputs = [node.outputs(0) for node in nodes]
    for output in outputs:
        print(output)
        assert output.shape == [B, N, P // 4]
        assert output.valmap == ValueMap(0, 1)