import cube.algorithm.ops.complex as complex
from cube.graph.tensor import IRFullTensor, ValueMap


def test_complex_toqkv_data_parallel():

    L = 64      # seq len
    N = 16      # batch
    E = 1024    # hiddend size = dim_head * num_head
    num_heads = 8
    dim_head = E // num_heads
    input = IRFullTensor(shape=[L, N, E], name='hidden').tosub()
    weight = IRFullTensor(shape=[3 * E, E], name='weight').tosub()

    semantic_op = complex.CubeComplexToQKV(
        signature='cube.runtime.function.complex.toqkv',
        inputs = [input, weight, num_heads]
    )
    semantic_op.infer_shape()

    qkv_dp = complex.CubeToQKVDataParallel(semantic_op)

    assert qkv_dp.chunk_num is None

    assert qkv_dp.satisfy(dict(chunk_num=8))
    assert not qkv_dp.satisfy(dict(chunk_num=32))

    nodes = qkv_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, complex.CubeComplexToQKV)
    
    inputs = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in inputs:
        print(input)
        assert input.shape == [L, N // 4, E]
    weights = [node.inputs(1) for node in nodes]

    print('weights:')
    for weight in weights:
        print(weight)
        assert weight.shape == [3 * E, E]

    sub_heads = [node.kwargs['num_heads'] for node in nodes]
    print('num_heads:')
    for nhead in sub_heads:
        assert nhead == 8
        print(nhead)

    outputs = [node.outputs() for node in nodes]
    print('outputs:')
    for output in outputs:
        q, k, v = output
        print('q:', q)
        print('k:', k)
        print('v:', v)
        assert q.shape == [L, N * num_heads // 4, dim_head]
        assert k.shape == [L, N * num_heads // 4, dim_head]
        assert v.shape == [L, N * num_heads // 4, dim_head]


def test_complex_toqkv_head_parallel():

    L = 64      # seq len
    N = 16      # batch
    E = 1024    # hiddend size = dim_head * num_head
    num_heads = 8
    dim_head = E // num_heads
    input = IRFullTensor(shape=[L, N, E], name='hidden').tosub()
    weight = IRFullTensor(shape=[3 * E, E], name='weight').tosub()

    semantic_op = complex.CubeComplexToQKV(
        signature='cube.runtime.function.complex.toqkv',
        inputs = [input, weight, num_heads]
    )
    semantic_op.infer_shape()

    qkv_hp = complex.CubeToQKVHeadParallel(semantic_op)

    assert qkv_hp.chunk_num is None

    assert qkv_hp.satisfy(dict(chunk_num=8))
    assert not qkv_hp.satisfy(dict(chunk_num=32))

    nodes = qkv_hp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, complex.CubeComplexToQKV)
    
    inputs = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in inputs:
        print(input)
        assert input.shape == [L, N, E]

    weights = [node.inputs(1) for node in nodes]
    print('weights:')
    for weight in weights:
        assert weight.shape == [3 * E // 4, E]
        print(weight)

    sub_heads = [node.kwargs['num_heads'] for node in nodes]
    print('sub_heads:')
    for nhead in sub_heads:
        assert nhead == num_heads // 4
        print(nhead)

    outputs = [node.outputs() for node in nodes]
    print('outputs:')
    for output in outputs:
        q, k, v = output
        print('q:', q)
        print('k:', k)
        print('v:', v)
        assert q.shape == [L, N * num_heads // 4, dim_head]
        assert k.shape == [L, N * num_heads // 4, dim_head]
        assert v.shape == [L, N * num_heads // 4, dim_head]


def test_complex_tril_mask_data_parallel():

    L = 64      # seq len
    N = 16      # batch
    num_heads = 8
    input = IRFullTensor(shape=[N * num_heads, L, L], name='hidden').tosub()
    
    semantic_op = complex.CubeComplexTrilMask(
        signature = 'cube.runtime.function.complex.trill_mask',
        inputs = [input, num_heads],
    )
    semantic_op.infer_shape()

    mask_dp = complex.CubeTrilMaskDataParallel(semantic_op)

    assert mask_dp.chunk_num is None

    assert mask_dp.satisfy(dict(chunk_num=8))
    assert not mask_dp.satisfy(dict(chunk_num=32))

    nodes = mask_dp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, complex.CubeComplexTrilMask)
    
    inputs = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in inputs:
        print(input)
        assert input.shape == [N * num_heads // 4, L, L]

    sub_heads = [node.kwargs['num_heads'] for node in nodes]
    print('num_heads:')
    for nhead in sub_heads:
        assert nhead == 8
        print(nhead)

    outputs = [node.outputs(0) for node in nodes]
    print('outputs:')
    for output in outputs:
        print(output)
        assert output.shape == [N * num_heads // 4, L, L]


def test_complex_tril_mask_head_parallel():

    L = 64      # seq len
    N = 16      # batch
    num_heads = 8
    input = IRFullTensor(shape=[N * num_heads, L, L], name='hidden').tosub()
    
    semantic_op = complex.CubeComplexTrilMask(
        signature = 'cube.runtime.function.complex.trill_mask',
        inputs = [input, num_heads],
    )
    semantic_op.infer_shape()

    mask_hp = complex.CubeTrilMaskHeadParallel(semantic_op)

    assert mask_hp.chunk_num is None

    assert mask_hp.satisfy(dict(chunk_num=8))
    assert not mask_hp.satisfy(dict(chunk_num=32))

    nodes = mask_hp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, complex.CubeComplexTrilMask)
    
    inputs = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in inputs:
        print(input)
        assert input.shape == [N * num_heads // 4, L, L]

    sub_heads = [node.kwargs['num_heads'] for node in nodes]
    print('num_heads:')
    for nhead in sub_heads:
        assert nhead == num_heads // 4
        print(nhead)

    outputs = [node.outputs(0) for node in nodes]
    print('outputs:')
    for output in outputs:
        print(output)
        assert output.shape == [N * num_heads // 4, L, L]


def test_complex_attn_view_data_parallel():

    L = 64      # seq len
    N = 16      # batch
    num_heads = 8
    dim_head = 128
    input = IRFullTensor(
        shape=[N * num_heads, L, dim_head], name='hidden').tosub()
    
    semantic_op = complex.CubeComplexAttnView(
        signature = 'cube.runtime.function.complex.trill_mask',
        inputs = [input, num_heads],
    )
    semantic_op.infer_shape()

    mask_hp = complex.CubeAttnViewDataParallel(semantic_op)

    assert mask_hp.chunk_num is None

    assert mask_hp.satisfy(dict(chunk_num=8))
    assert not mask_hp.satisfy(dict(chunk_num=32))

    nodes = mask_hp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, complex.CubeComplexAttnView)
    
    inputs = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in inputs:
        print(input)
        assert input.shape == [N * num_heads // 4, L, dim_head]

    sub_heads = [node.kwargs['num_heads'] for node in nodes]
    print('num_heads:')
    for nhead in sub_heads:
        assert nhead == num_heads
        print(nhead)

    outputs = [node.outputs(0) for node in nodes]
    print('outputs:')
    for output in outputs:
        print(output)
        assert output.shape == [L, N // 4, num_heads * dim_head]


def test_complex_attn_view_head_parallel():

    L = 64      # seq len
    N = 16      # batch
    num_heads = 8
    dim_head = 128
    input = IRFullTensor(
        shape=[N * num_heads, L, dim_head], name='hidden').tosub()
    
    semantic_op = complex.CubeComplexAttnView(
        signature = 'cube.runtime.function.complex.trill_mask',
        inputs = [input, num_heads],
    )
    semantic_op.infer_shape()

    mask_hp = complex.CubeAttnViewHeadParallel(semantic_op)

    assert mask_hp.chunk_num is None

    assert mask_hp.satisfy(dict(chunk_num=8))
    assert not mask_hp.satisfy(dict(chunk_num=32))

    nodes = mask_hp.instantiate(semantic_op, dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, complex.CubeComplexAttnView)
    
    inputs = [node.inputs(0) for node in nodes]
    print('inputs:')
    for input in inputs:
        print(input)
        assert input.shape == [N * num_heads // 4, L, dim_head]

    sub_heads = [node.kwargs['num_heads'] for node in nodes]
    print('num_heads:')
    for nhead in sub_heads:
        assert nhead == num_heads // 4
        print(nhead)

    outputs = [node.outputs(0) for node in nodes]
    print('outputs:')
    for output in outputs:
        print(output)
        assert output.shape == [L, N, num_heads * dim_head // 4]
