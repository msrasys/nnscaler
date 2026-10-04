"""Merged restore must match the full padded bucket at every ZeRO boundary."""
from types import SimpleNamespace

import pytest
import torch

from nnscaler.parallel import _construct_optim_state_zero


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('shards', [1, 2, 4, 8])
@pytest.mark.parametrize('value_chunks', [1, 2, 3])
@pytest.mark.parametrize('step', [False, True])
def test_zero_restore_boundaries_padding_tp_and_source_isolation(dtype, shards, value_chunks, step):
    # Include a noncontiguous tensor-parallel slice. Sizes and alignment make
    # shard boundaries fall inside data, inside padding, and on parameter ends.
    sizes, offsets = [6, 13, 3], [0, 8, 24]
    params = [torch.nn.Parameter(torch.empty(n, dtype=dtype)) for n in sizes]
    outside = torch.nn.Parameter(torch.empty(2, dtype=dtype))
    named = [(f'p{i}_{i}', p) for i, p in enumerate(params + [outside])]
    fullmap, source = {}, {}
    for i, (_, p) in enumerate(named):
        value = torch.arange(p.numel() * 2, dtype=dtype).reshape(-1, 2) + i
        fullmap[named[i][0]] = SimpleNamespace(slicers=(slice(None), 0), val_chunks=value_chunks)
        source[f'p{i}'] = {'exp_avg': value, 'exp_avg_sq': value + 3}
        if step:
            source[f'p{i}']['step'] = torch.tensor(20.)
    original = {n: {k: v.clone() for k, v in s.items()} for n, s in source.items()}
    bucket = SimpleNamespace(params=params, _contiguous_params=torch.empty(32))
    positions = {id(p): offset for p, offset in zip(params, offsets)}
    reducer = SimpleNamespace(params=params, buckets=[bucket], get_param_info=lambda p:
                              SimpleNamespace(bucket_param_buffer_start=positions[id(p)]))
    module = SimpleNamespace(
        dist_param_map={f'p{i}': f'p{i}' for i in range(4)}, fullmap=fullmap,
        named_parameters=lambda: iter(named), parameters=lambda: iter(params + [outside]),
        reducers=[reducer], parameters_for_optimizer=lambda: [torch.empty(32 // shards), outside],
    )
    expected = {}
    for key in ['exp_avg', 'exp_avg_sq']:
        flat = torch.zeros(32, dtype=dtype)
        for i, (size, offset) in enumerate(zip(sizes, offsets)):
            flat[offset:offset + size] = source[f'p{i}'][key][:, 0] / value_chunks
        expected[key] = flat
    for rank in range(shards):
        module._get_zero_subranks = lambda r: (rank, list(range(shards)))
        result = _construct_optim_state_zero(module, source)
        assert list(result[0]) == ['exp_avg', 'exp_avg_sq'] + (['step'] if step else [])
        for key, flat in expected.items():
            torch.testing.assert_close(result[0][key], flat.chunk(shards)[rank], rtol=0, atol=0)
            torch.testing.assert_close(result[1][key], source['p3'][key][:, 0] / value_chunks, rtol=0, atol=0)
            # Restored state owns its storage; a subsequent optimizer update
            # must never modify the mmap-backed checkpoint or another rank.
            result[0][key].fill_(-1)
            result[1][key].fill_(-2)
        assert ('step' in result[0]) == step
        if step:
            assert result[0]['step'].item() == 20
    for name, state in source.items():
        for key, value in state.items():
            torch.testing.assert_close(value, original[name][key], rtol=0, atol=0)


@pytest.mark.parametrize('source_dtype', [torch.bfloat16, torch.int64])
def test_mixed_state_dtype_preserves_rounding_before_bucket_conversion(source_dtype):
    params = [torch.nn.Parameter(torch.empty(6)) for _ in range(2)]
    named = [(f'p{i}_{i}', p) for i, p in enumerate(params)]
    source = {f'p{i}': {'value': torch.arange(6, dtype=dtype) + 1, 'step': None}
              for i, dtype in enumerate([torch.float32, source_dtype])}
    bucket = SimpleNamespace(params=params, _contiguous_params=torch.empty(12))
    offsets = {id(p): i * 6 for i, p in enumerate(params)}
    reducer = SimpleNamespace(params=params, buckets=[bucket], get_param_info=lambda p:
                              SimpleNamespace(bucket_param_buffer_start=offsets[id(p)]))
    module = SimpleNamespace(
        dist_param_map={f'p{i}': f'p{i}' for i in range(2)},
        fullmap={name: SimpleNamespace(slicers=(slice(None),), val_chunks=3) for name, _ in named},
        named_parameters=lambda: iter(named), parameters=lambda: iter(params),
        reducers=[reducer], parameters_for_optimizer=lambda: [torch.empty(12)],
        _get_zero_subranks=lambda r: (0, [0]),
    )
    expected = torch.cat([(source[f'p{i}']['value'] / 3).float() for i in range(2)])
    result = _construct_optim_state_zero(module, source)
    assert 'step' not in result[0]
    torch.testing.assert_close(result[0]['value'], expected, rtol=0, atol=0)
