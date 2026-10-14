"""Merged restore must match the full padded bucket at every ZeRO boundary."""
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from nnscaler.parallel import _construct_optim_state_zero, _copy_optim_state_slice


class AllocationSizes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.sizes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        inputs = {value.untyped_storage().data_ptr()
                  for value in tree_leaves((args, kwargs)) if isinstance(value, torch.Tensor)}
        result = func(*args, **kwargs)
        for value in tree_leaves(result):
            if isinstance(value, torch.Tensor) and value.untyped_storage().data_ptr() not in inputs:
                self.sizes.append(value.untyped_storage().nbytes())
        return result


@pytest.mark.parametrize('layout', ['contiguous', 'cropped', 'transposed', 'permuted', 'expanded'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.int64])
def test_logical_slice_values_rounding_and_allocation(layout, dtype):
    base = torch.arange(7 * 9 * 11).reshape(7, 9, 11).to(dtype)
    source = {
        'contiguous': base,
        'cropped': base[1:6, 1:8, 2:9],
        'transposed': base[0].t(),
        'permuted': base.permute(2, 0, 1),
        'expanded': base[:1].expand(7, -1, -1),
    }[layout]
    expected = (source / 3).flatten().float()
    original = source.clone()
    size = source.numel()
    for offset, count in [(0, size), (1, 1), (1, size - 2), (size // 3, size // 2), (size - 1, 1), (size, 0)]:
        destination = torch.empty(count, dtype=torch.float32)
        with AllocationSizes() as allocations:
            _copy_optim_state_slice(destination, source, offset, 3)
        torch.testing.assert_close(destination, expected[offset:offset + count], rtol=0, atol=0)
        assert max(allocations.sizes, default=0) <= count * destination.element_size()
    torch.testing.assert_close(source, original, rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_partial_tp_restore_does_not_allocate_the_full_parameter(dtype):
    # Cropped matrix rows cannot be flattened as a view. ZeRO boundaries also
    # cut through rows, so restoring one shard must handle partial edge rows.
    source = torch.arange(32 * 80, dtype=torch.float32).reshape(32, 80).to(dtype)
    slicers = (slice(None), slice(5, 69))
    param = torch.nn.Parameter(torch.empty(32, 64, dtype=dtype))
    opt_param = torch.empty(260, dtype=dtype)
    bucket = SimpleNamespace(params=[param], _contiguous_params=torch.empty(2080))
    reducer = SimpleNamespace(params=[param], buckets=[bucket], get_param_info=lambda p:
                              SimpleNamespace(bucket_param_buffer_start=8))
    module = SimpleNamespace(
        dist_param_map={'p': 'original'},
        fullmap={'p_0': SimpleNamespace(slicers=slicers, val_chunks=3)},
        named_parameters=lambda: iter([('p_0', param)]), parameters=lambda: iter([param]),
        reducers=[reducer], parameters_for_optimizer=lambda: [opt_param],
    )
    expected = torch.zeros(2080, dtype=dtype)
    expected[8:2056] = (source[slicers] / 3).flatten()
    original = source.clone()
    for rank in range(8):
        module._get_zero_subranks = lambda r: (rank, list(range(8)))
        with AllocationSizes() as allocations:
            result = _construct_optim_state_zero(module, {'original': {'exp_avg': source}})
        torch.testing.assert_close(result[0]['exp_avg'], expected.chunk(8)[rank], rtol=0, atol=0)
        assert max(allocations.sizes, default=0) <= 260 * source.element_size()
    torch.testing.assert_close(source, original, rtol=0, atol=0)


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
