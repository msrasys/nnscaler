import cube.algorithm.ops.complex as complex
from cube.graph.tensor import IRFullTensor, ValueMap


def test_complex_toqkv_data_parallel():

    L = 64      # seq len
    N = 16      # batch
    E = 1024    # hiddend size = dim_head * num_head
    num_head = 8
    dim_head = E // num_head
    input = IRFullTensor(shape=[L, N, E], name='hidden').tosub()
    weight = IRFullTensor(shape=[3 * E, E], name='weight').tosub()

    semantic_op = complex.CubeComplexToQKV(
        signature='cube.runtime.function.complex.toqkv',
        inputs = [input, weight, num_head]
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

    sub_heads = [node.kwargs['num_head'] for node in nodes]
    print('num_head:')
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
        assert q.shape == [L, N * num_head // 4, dim_head]
        assert k.shape == [L, N * num_head // 4, dim_head]
        assert v.shape == [L, N * num_head // 4, dim_head]


def test_complex_toqkv_head_parallel():

    L = 64      # seq len
    N = 16      # batch
    E = 1024    # hiddend size = dim_head * num_head
    num_head = 8
    dim_head = E // num_head
    input = IRFullTensor(shape=[L, N, E], name='hidden').tosub()
    weight = IRFullTensor(shape=[3 * E, E], name='weight').tosub()

    semantic_op = complex.CubeComplexToQKV(
        signature='cube.runtime.function.complex.toqkv',
        inputs = [input, weight, num_head]
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

    sub_heads = [node.kwargs['num_head'] for node in nodes]
    print('sub_heads:')
    for nhead in sub_heads:
        assert nhead == num_head // 4
        print(nhead)

    outputs = [node.outputs() for node in nodes]
    print('outputs:')
    for output in outputs:
        q, k, v = output
        print('q:', q)
        print('k:', k)
        print('v:', v)
        assert q.shape == [L, N * num_head // 4, dim_head]
        assert k.shape == [L, N * num_head // 4, dim_head]
        assert v.shape == [L, N * num_head // 4, dim_head]


def test_complex_tril_mask_data_parallel():

    L = 64      # seq len
    N = 16      # batch
    num_head = 8
    input = IRFullTensor(shape=[N * num_head, L, L], name='hidden').tosub()
    
    semantic_op = complex.CubeComplexTrilMask(
        signature = 'cube.runtime.function.complex.trill_mask',
        inputs = [input, num_head],
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
        assert input.shape == [N * num_head // 4, L, L]

    sub_heads = [node.kwargs['num_head'] for node in nodes]
    print('num_head:')
    for nhead in sub_heads:
        assert nhead == 8
        print(nhead)

    outputs = [node.outputs(0) for node in nodes]
    print('outputs:')
    for output in outputs:
        print(output)
        assert output.shape == [N * num_head // 4, L, L]


def test_complex_tril_mask_head_parallel():

    L = 64      # seq len
    N = 16      # batch
    num_head = 8
    input = IRFullTensor(shape=[N * num_head, L, L], name='hidden').tosub()
    
    semantic_op = complex.CubeComplexTrilMask(
        signature = 'cube.runtime.function.complex.trill_mask',
        inputs = [input, num_head],
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
        assert input.shape == [N * num_head // 4, L, L]

    sub_heads = [node.kwargs['num_head'] for node in nodes]
    print('num_head:')
    for nhead in sub_heads:
        assert nhead == num_head // 4
        print(nhead)

    outputs = [node.outputs(0) for node in nodes]
    print('outputs:')
    for output in outputs:
        print(output)
        assert output.shape == [N * num_head // 4, L, L]


def test_complex_attn_view_data_parallel():

    L = 64      # seq len
    N = 16      # batch
    num_head = 8
    dim_head = 128
    input = IRFullTensor(
        shape=[N * num_head, L, dim_head], name='hidden').tosub()
    
    semantic_op = complex.CubeComplexAttnView(
        signature = 'cube.runtime.function.complex.trill_mask',
        inputs = [input, num_head],
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
        assert input.shape == [N * num_head // 4, L, dim_head]

    sub_heads = [node.kwargs['num_head'] for node in nodes]
    print('num_head:')
    for nhead in sub_heads:
        assert nhead == num_head
        print(nhead)

    outputs = [node.outputs(0) for node in nodes]
    print('outputs:')
    for output in outputs:
        print(output)
        assert output.shape == [L, N // 4, num_head * dim_head]


def test_complex_attn_view_head_parallel():

    L = 64      # seq len
    N = 16      # batch
    num_head = 8
    dim_head = 128
    input = IRFullTensor(
        shape=[N * num_head, L, dim_head], name='hidden').tosub()
    
    semantic_op = complex.CubeComplexAttnView(
        signature = 'cube.runtime.function.complex.trill_mask',
        inputs = [input, num_head],
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
        assert input.shape == [N * num_head // 4, L, dim_head]

    sub_heads = [node.kwargs['num_head'] for node in nodes]
    print('num_head:')
    for nhead in sub_heads:
        assert nhead == num_head // 4
        print(nhead)

    outputs = [node.outputs(0) for node in nodes]
    print('outputs:')
    for output in outputs:
        print(output)
        assert output.shape == [L, N, num_head * dim_head // 4]


def test_complex_self_attention_head_parallel():
    L = 64      # seq len
    N = 16      # batch
    num_head = 8
    dim_head = 128
    E = num_head * dim_head

    input = IRFullTensor(
        shape=[L, N, E], name='hidden').tosub()
    w_qkv = IRFullTensor(
        shape=[3 * num_head * dim_head, num_head * dim_head], name='wqkv').tosub()
    w_out = IRFullTensor(
        shape=[num_head * dim_head, num_head * dim_head], name='wout').tosub()

    semantic_op = complex.CubeComplexSelfAttention(
        signature = 'cube.runtime.function.complex.self_attn',
        inputs = [input, w_qkv, w_out, num_head, dim_head, 0.5],
    )
    semantic_op.infer_shape()

    op_head = complex.CubeSelfAttentionHeadParallel(semantic_op)

    assert op_head.satisfy(config=dict(chunk_num=8))
    assert not op_head.satisfy(config=dict(chunk_num=16))
    
    nodes = op_head.instantiate(semantic_op, config=dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, complex.CubeComplexSelfAttention)
    
    for idx, node in enumerate(nodes):
        assert node.outputs(0).shape == [L, N, E]
        assert node.outputs(0).val_map == ValueMap(idx, 4)
        assert node.kwargs['num_head'] == num_head // 4
        assert node.inputs(0).shape == [L, N, E]
        assert node.inputs(1).shape == [3 * E // 4, E]
        assert node.inputs(2).shape == [E, E // 4]


def test_complex_self_attention_data_parallel():
    L = 64      # seq len
    N = 16      # batch
    num_head = 8
    dim_head = 128
    E = num_head * dim_head

    input = IRFullTensor(
        shape=[L, N, E], name='hidden').tosub()
    w_qkv = IRFullTensor(
        shape=[3 * num_head * dim_head, num_head * dim_head], name='wqkv').tosub()
    w_out = IRFullTensor(
        shape=[num_head * dim_head, num_head * dim_head], name='wout').tosub()

    semantic_op = complex.CubeComplexSelfAttention(
        signature = 'cube.runtime.function.complex.self_attn',
        inputs = [input, w_qkv, w_out, num_head, dim_head, 0.5],
    )
    semantic_op.infer_shape()

    op_head = complex.CubeSelfAttentionDataParallel(semantic_op)

    assert op_head.satisfy(config=dict(chunk_num=8))
    assert not op_head.satisfy(config=dict(chunk_num=32))
    
    nodes = op_head.instantiate(semantic_op, config=dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, complex.CubeComplexSelfAttention)
    
    for idx, node in enumerate(nodes):
        assert node.outputs(0).shape == [L, N // 4, E]
        assert node.outputs(0).val_map == ValueMap(0, 1)
        assert node.kwargs['num_head'] == num_head
        assert node.inputs(0).shape == [L, N // 4, E]
        assert node.inputs(1).shape == [3 * E, E]
        assert node.inputs(2).shape == [E, E]


def test_complex_feedforward_tensor_parallel():
    L = 64      # seq len
    N = 16      # batch
    E = 1024

    input = IRFullTensor(
        shape=[L, N, E], name='hidden').tosub()
    w_proj1 = IRFullTensor(
        shape=[4 * E, E], name='proj1').tosub()
    w_bias1 = IRFullTensor(
        shape=[4 * E,], name='bias1').tosub()
    w_proj2 = IRFullTensor(
        shape=[E, 4 * E], name='proj2').tosub()
    w_bias2 = IRFullTensor(
        shape=[E,], name='bias2').tosub()

    semantic_op = complex.CubeComplexFeedForward(
        signature = 'cube.runtime.function.complex.feedforward',
        inputs = [input, w_proj1, w_bias1, w_proj2, w_bias2],
    )
    semantic_op.infer_shape()

    op_head = complex.CubeFeedForwardTensorParallel(semantic_op)

    assert op_head.satisfy(config=dict(chunk_num=8))
    assert op_head.satisfy(config=dict(chunk_num=32))
    
    nodes = op_head.instantiate(semantic_op, config=dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, complex.CubeComplexFeedForward)
    
    for idx, node in enumerate(nodes):
        assert node.outputs(0).shape == [L, N, E]
        assert node.outputs(0).val_map == ValueMap(idx, 4)
        assert node.inputs(0).shape == [L, N, E]
        assert node.inputs(1).shape == [4 * E // 4, E]
        assert node.inputs(2).shape == [4 * E // 4,]
        assert node.inputs(3).shape == [E, 4 * E // 4]
        assert node.inputs(4).shape == [E,]
        assert node.inputs(4).val_map == ValueMap(idx, 4)


def test_complex_feedforward_data_parallel():
    L = 64      # seq len
    N = 16      # batch
    E = 1024

    input = IRFullTensor(
        shape=[L, N, E], name='hidden').tosub()
    w_proj1 = IRFullTensor(
        shape=[4 * E, E], name='proj1').tosub()
    w_bias1 = IRFullTensor(
        shape=[4 * E,], name='bias1').tosub()
    w_proj2 = IRFullTensor(
        shape=[E, 4 * E], name='proj2').tosub()
    w_bias2 = IRFullTensor(
        shape=[E,], name='bias2').tosub()

    semantic_op = complex.CubeComplexFeedForward(
        signature = 'cube.runtime.function.complex.feedforward',
        inputs = [input, w_proj1, w_bias1, w_proj2, w_bias2],
    )
    semantic_op.infer_shape()

    op_head = complex.CubeFeedForwardDataParallel(semantic_op)

    assert op_head.satisfy(config=dict(chunk_num=8))
    assert not op_head.satisfy(config=dict(chunk_num=32))
    
    nodes = op_head.instantiate(semantic_op, config=dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, complex.CubeComplexFeedForward)
    
    for idx, node in enumerate(nodes):
        assert node.outputs(0).shape == [L, N // 4, E]
        assert node.outputs(0).val_map == ValueMap(0, 1)
        assert node.inputs(0).shape == [L, N // 4, E]
        assert node.inputs(1).shape == [4 * E, E]
        assert node.inputs(2).shape == [4 * E,]
        assert node.inputs(3).shape == [E, 4 * E]
        assert node.inputs(4).shape == [E,]


def test_embed_shard_parallel():
    L = 64      # seq len
    N = 16      # batch
    vocab = 50304
    E = 1024

    ids = IRFullTensor(shape=[L, N], name='hidden').tosub()
    weight = IRFullTensor(shape=[vocab, E], name='hidden').tosub()
    start = 0
    stop = vocab

    semantic_op = complex.CubeComplexEmbedding(
        signature = 'cube.runtime.function.complex.embedding',
        inputs = [ids, weight, start, stop]
    )
    semantic_op.infer_shape()

    assert semantic_op.outputs(0).shape == [L, N, E]

    op_shard = complex.CubeEmbedShardingParallel(semantic_op)
    
    assert op_shard.satisfy(config=dict(chunk_num=8))
    assert op_shard.satisfy(config=dict(chunk_num=32))
    assert not op_shard.satisfy(config=dict(chunk_num=256))

    nodes = op_shard.instantiate(semantic_op, config=dict(chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, complex.CubeComplexEmbedding)
    
    start = semantic_op.kwargs['start']
    stop = semantic_op.kwargs['stop']
    shard = (stop - start) // 4
    for idx, node in enumerate(nodes):
        assert node.outputs(0).shape == [L, N, E]
        assert node.outputs(0).val_map == ValueMap(idx, 4)
        assert node.inputs(0).shape == [L, N]
        assert node.inputs(1).shape == [vocab // 4, E]
        assert node.kwargs['start'] == start + idx * shard
        assert node.kwargs['stop'] == start + (idx + 1) * shard


def test_embed_shard_parallel():
    L = 64      # seq len
    N = 16      # batch
    vocab = 50304
    E = 1024

    ids = IRFullTensor(shape=[L, N], name='hidden').tosub()
    weight = IRFullTensor(shape=[vocab, E], name='hidden').tosub()
    start = 0
    stop = vocab

    semantic_op = complex.CubeComplexEmbedding(
        signature = 'cube.runtime.function.complex.embedding',
        inputs = [ids, weight, start, stop]
    )
    semantic_op.infer_shape()

    assert semantic_op.outputs(0).shape == [L, N, E]

    op_shard = complex.CubeEmbedDataParallel(semantic_op)
    
    assert op_shard.satisfy(config=dict(dim=1, chunk_num=8))
    assert not op_shard.satisfy(config=dict(dim=1, chunk_num=32))

    nodes = op_shard.instantiate(semantic_op, config=dict(dim=1, chunk_num=4))
    assert len(nodes) == 4
    for node in nodes:
        assert isinstance(node, complex.CubeComplexEmbedding)
    
    start = semantic_op.kwargs['start']
    stop = semantic_op.kwargs['stop']
    for idx, node in enumerate(nodes):
        assert node.outputs(0).shape == [L, N // 4, E]
        assert node.outputs(0).val_map == ValueMap(0, 1)
        assert node.inputs(0).shape == [L, N // 4]
        assert node.inputs(1).shape == [vocab, E]
        assert node.kwargs['start'] == start
        assert node.kwargs['stop'] == stop
