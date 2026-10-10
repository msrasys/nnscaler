#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import random
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from nnscaler.parallel import ParamInitStrategy
from nnscaler.runtime.initialization import (
    DeferredInitialization, create_init_module, create_shard_init_weights, capture_init_weights,
)
from nnscaler.runtime.module import AttrMeta, ParallelModule
from tests.parallel_module.common import assert_equal


class RandomInitModule(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(8))
        self.register_buffer('python_numpy', torch.tensor([random.random(), np.random.rand()]))
        self.register_buffer('device_random', torch.rand(8, device=device))


def _rng_state():
    numpy_state = np.random.get_state()
    return (
        random.getstate(),
        (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_seeded_initialization_restores_rng(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    before = _rng_state()
    first = create_init_module(
        RandomInitModule, lambda: RandomInitModule(device), torch.float64, seed=17,
    )
    assert_equal(before, _rng_state())
    random.random()
    np.random.rand(13)
    torch.rand(19)
    torch.rand(23, device=device)
    before = _rng_state()
    second = create_init_module(
        RandomInitModule, lambda: RandomInitModule(device), torch.float64, seed=17,
    )
    assert_equal(before, _rng_state())
    assert_equal(first.state_dict(), second.state_dict())
    assert all(t.dtype == torch.float64 for t in first.state_dict().values())
    third = create_init_module(RandomInitModule, None, torch.float64, seed=18)
    assert not torch.equal(first.weight, third.weight)

    def failing_factory():
        RandomInitModule(device)
        raise RuntimeError('initialization failed')

    before = _rng_state()
    with pytest.raises(RuntimeError, match='initialization failed'):
        create_init_module(RandomInitModule, failing_factory, None, seed=19)
    assert_equal(before, _rng_state())


def test_unseeded_initialization_preserves_existing_behavior():
    cpu_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    create_init_module(RandomInitModule, None, None)
    assert not torch.equal(cpu_state, torch.get_rng_state())
    assert numpy_state[2] != np.random.get_state()[2]
    assert python_state != random.getstate()


class DependentBufferModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(4))
        torch.nn.init.uniform_(self.weight, -1, 1)
        self.register_buffer('copy', self.weight.detach().clone(), persistent=False)


def _local_module(strategy):
    class LocalModule(ParallelModule, skip_init=True):
        pass

    module = object.__new__(LocalModule)
    torch.nn.Module.__init__(module)
    module.compute_config = SimpleNamespace(param_init_strategy=strategy)
    module.register_parameter('weight_local', torch.nn.Parameter(torch.empty(2)))
    module.register_buffer('copy_local', torch.empty(2), persistent=False)
    module._fullmap = {
        name: AttrMeta(
            tid=idx, is_param=idx == 0, orig_name=original,
            shape=(4,), slicers=(slice(1, 3),), val_chunks=1,
            dtype=torch.float32, sub_shape=(2,),
        )
        for idx, (name, original) in enumerate([('weight_local', 'weight'), ('copy_local', 'copy')])
    }
    LocalModule._init_module_fn = staticmethod(partial(
        create_init_module, DependentBufferModule, None, None, seed=1234,
    ))
    return module


@pytest.mark.parametrize('strategy', [ParamInitStrategy.FULL, ParamInitStrategy.SHARD])
@pytest.mark.parametrize('val_chunks', [1, 2])
def test_source_strategy_preserves_buffer_dependency_on_init_and_resume(tmp_path, strategy, val_chunks):
    module = _local_module(strategy)
    module.module_dir = tmp_path
    module._fullmap['weight_local'].val_chunks = val_chunks
    module._fullmap['copy_local'].val_chunks = val_chunks
    if strategy == ParamInitStrategy.FULL:
        source = type(module)._init_module_fn()
        torch.save({module.fullmap['copy_local'].tid: source.copy}, tmp_path / 'npbuffer.pt')
        contents = list(module._iter_np_buffer_content(str(tmp_path / 'npbuffer.pt')))
        assert len(contents) == 1
        attr_name, content = contents[0]
        assert attr_name == 'copy_local'
        torch.testing.assert_close(content, source.copy[1:3] / val_chunks, rtol=0, atol=0)
    with patch.object(module, 'load_np_buffer_content', wraps=module.load_np_buffer_content) as load, \
            patch.object(module, 'check_np_buffer_content', wraps=module.check_np_buffer_content) as check:
        module._post_init(build_buckets=False)
        load.assert_not_called()
        if strategy == ParamInitStrategy.FULL:
            check.assert_called_once_with(str(tmp_path / 'npbuffer.pt'))
        else:
            check.assert_not_called()
        torch.testing.assert_close(module.weight_local, module.copy_local, rtol=0, atol=0)
        initial_buffer = module.copy_local.clone()
        with torch.no_grad():
            module.weight_local.fill_(9)
            module.copy_local.fill_(0)
        if strategy == ParamInitStrategy.FULL:
            with patch.object(type(module), '_init_module_fn', side_effect=AssertionError('resume constructor')):
                module._post_init(init_params=False, build_buckets=False)
            load.assert_called_once_with(str(tmp_path / 'npbuffer.pt'))
            assert check.call_count == 1
        else:
            module._post_init(init_params=False, build_buckets=False)
            load.assert_not_called()
        assert torch.equal(module.weight_local, torch.full((2,), 9.0))
        assert torch.equal(module.copy_local, initial_buffer)
        assert module.non_presistent_buffers_inited


@pytest.mark.parametrize('difference', ['value', 'signed_zero', 'dtype', 'shape', 'missing'])
def test_full_initialization_checks_saved_buffers_without_overwriting(tmp_path, difference):
    module = _local_module(ParamInitStrategy.FULL)
    module.module_dir = tmp_path
    module._fullmap['copy_local'].val_chunks = 2
    source = DependentBufferModule()
    source.copy.zero_()
    saved = source.copy.clone()
    if difference == 'value':
        saved[1] = 1
    elif difference == 'signed_zero':
        saved = -saved
    elif difference == 'dtype':
        saved = saved.double()
    elif difference == 'shape':
        saved = saved.reshape(4, 1)
    torch.save({1: saved} if difference != 'missing' else {}, tmp_path / 'npbuffer.pt')
    with pytest.raises(RuntimeError, match='differs from|not found'):
        module._post_init(init_module=source, build_buckets=False)
    assert torch.equal(module.copy_local, source.copy[1:3] / 2)
    assert not torch.signbit(module.copy_local).any()


def test_shard_instance_bypasses_attached_hook_on_init_and_resume():
    module = _local_module(ParamInitStrategy.SHARD)
    source = DependentBufferModule()
    with torch.no_grad():
        source.weight.fill_(6)
        source.copy.fill_(7)
    module._fullmap['weight_local'].val_chunks = 2
    with patch.object(type(module), '_shard_init_fn', side_effect=AssertionError('hook')), \
            patch.object(type(module), '_init_module_fn', side_effect=AssertionError('constructor')):
        module._post_init(init_module=source, build_buckets=False)
        assert torch.equal(module.weight_local, torch.full((2,), 3.0))
        assert torch.equal(module.copy_local, torch.full((2,), 7.0))
        with torch.no_grad():
            module.weight_local.fill_(9)
            source.copy.fill_(8)
        module._post_init(init_module=source, init_params=False, build_buckets=False)
        assert torch.equal(module.weight_local, torch.full((2,), 9.0))
        assert torch.equal(module.copy_local, torch.full((2,), 8.0))


@pytest.mark.parametrize('strategy', [ParamInitStrategy.FULL, ParamInitStrategy.SHARD])
@pytest.mark.parametrize('value', [torch.ones(3), torch.ones(4, dtype=torch.float64), 42])
def test_source_initializer_validates_tensor_metadata(strategy, value):
    module = _local_module(strategy)
    source = torch.nn.Module()
    source.weight = value
    with pytest.raises(RuntimeError, match='Invalid initialization tensor for weight'):
        module._post_init(init_module=source, build_buckets=False)


def test_capture_initializer_streams_into_model_and_releases_full_sources():
    from torch.multiprocessing.reductions import StorageWeakRef

    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(24.0).view(4, 6))
            self.tied = self.weight
            self.register_buffer('transposed', self.weight.detach().t())
            self.register_buffer('scalar', torch.full((), 0.5))
            self.unused = torch.nn.Parameter(torch.ones(17))

    specs = [
        ('first', 'weight', (4, 6), (slice(0, 2), slice(1, 4)), (2, 3)),
        ('second', 'tied', (4, 6), (slice(2, 4), slice(2, 5)), (2, 3)),
        ('view', 'transposed', (6, 4), (slice(1, 4), slice(1, 3)), (3, 2)),
        ('scalar', 'scalar', (), (), ()),
    ]
    attrs = {
        name: AttrMeta(
            tid=i, is_param=i < 2, orig_name=original, shape=shape, slicers=slices,
            val_chunks=1, dtype=torch.float32, sub_shape=local_shape,
        )
        for i, (name, original, shape, slices, local_shape) in enumerate(specs)
    }
    calls = []
    full_storages = []
    materialize = DeferredInitialization.materialize

    def record(capture, tensor):
        assert all(storage.expired() for storage in full_storages)
        calls.append(tuple(tensor.shape))
        value = materialize(capture, tensor)
        full_storages.append(StorageWeakRef(value.untyped_storage()))
        return value

    module = _local_module(ParamInitStrategy.SHARD)
    del module.weight_local
    del module.copy_local
    module._fullmap = attrs
    for name, meta in attrs.items():
        value = torch.empty(meta.sub_shape, dtype=meta.dtype)
        if meta.is_param:
            module.register_parameter(name, torch.nn.Parameter(value))
        else:
            module.register_buffer(name, value, persistent=False)
    type(module)._init_module_fn = staticmethod(partial(create_init_module, Source, None, None, seed=17))
    with patch.object(DeferredInitialization, 'materialize', record):
        module._post_init(build_buckets=False)
    assert calls == [(4, 6), (6, 4), ()]
    assert all(storage.expired() for storage in full_storages)
    reference = Source()
    # note first/second/view(weight/tied/transposed) are not shared the same storage here.
    for name, meta in attrs.items():
        value = getattr(module, name)
        assert value.shape == meta.sub_shape
        assert value.dtype == meta.dtype and value.device.type == 'cpu'
        assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
        torch.testing.assert_close(value, getattr(reference, meta.orig_name)[meta.slicers])


def test_empty_capture_initializer_skips_constructor():
    def fail():
        raise AssertionError('constructor invoked')

    assert list(capture_init_weights({}, module_fn=fail)) == []


def test_user_shard_initializer_receives_only_required_attributes():
    module = _local_module(ParamInitStrategy.SHARD)
    calls = []

    class CustomModule(torch.nn.Module):
        def __init__(self):
            raise AssertionError('full constructor invoked')

        @staticmethod
        def __shard__init__(attrs):
            calls.append(dict(attrs))
            return {name: torch.full(meta.sub_shape, 7.0, dtype=meta.dtype)
                    for name, meta in attrs.items()}

    type(module)._shard_init_fn = staticmethod(partial(
        create_shard_init_weights, CustomModule, seed=1234,
    ))
    module._fullmap['weight_local'].val_chunks = 2
    module._post_init(build_buckets=False)
    assert calls[0] == module.fullmap
    assert torch.equal(module.weight_local, torch.full((2,), 3.5))
    assert torch.equal(module.copy_local, torch.full((2,), 7.0))
    with torch.no_grad():
        module.weight_local.fill_(9)
        module.copy_local.fill_(0)
    module._post_init(init_params=False, build_buckets=False)
    assert set(calls[1]) == {'copy_local'}
    assert torch.equal(module.weight_local, torch.full((2,), 9.0))
    assert torch.equal(module.copy_local, torch.full((2,), 7.0))


@pytest.mark.parametrize('result,message', [
    (None, 'not iterable'),
    ({}, 'missing attributes'),
    ({'weight_local': 42}, 'Invalid'),
    ({'weight_local': torch.ones(4), 'copy_local': torch.ones(2)}, 'expected shape'),
    ({'weight_local': torch.ones(2, dtype=torch.float64), 'copy_local': torch.ones(2)}, 'dtype'),
    ({'weight_local': torch.ones(2, device='meta'), 'copy_local': torch.ones(2)}, 'Invalid'),
    ([('weight_local', torch.ones(2)), ('weight_local', torch.ones(2))], 'duplicate attribute'),
    ([('weight_local', torch.ones(2))], 'missing attributes'),
    ([('extra', torch.ones(2))], 'unexpected attributes'),
    ([([], torch.ones(2))], 'unexpected attributes'),
    ([('weight_local',)], 'pairs'),
    (['weight_local'], 'pairs'),
])
def test_user_shard_initializer_validates_local_results(result, message):
    module = _local_module(ParamInitStrategy.SHARD)

    class CustomModule(torch.nn.Module):
        __shard__init__ = staticmethod(lambda attrs: result)

    type(module)._shard_init_fn = staticmethod(partial(create_shard_init_weights, CustomModule, seed=17))
    with pytest.raises(TypeError if result is None else RuntimeError, match=message):
        module._post_init(build_buckets=False)


@pytest.mark.parametrize('streaming', [False, True])
def test_shard_initializer_seed_and_failure_restore_rng(streaming):
    class CustomModule(torch.nn.Module):
        fail = False

        @staticmethod
        def __shard__init__(attrs):
            def generate():
                value = torch.rand(4) + random.random() + np.random.rand()
                if CustomModule.fail:
                    raise RuntimeError('custom initialization failed')
                yield 'value', value
            return generate() if streaming else dict(generate())

    before = _rng_state()
    first = dict(create_shard_init_weights(CustomModule, {}, seed=17))
    assert_equal(before, _rng_state())
    torch.rand(11)
    random.random()
    np.random.rand(3)
    second = dict(create_shard_init_weights(CustomModule, {}, seed=17))
    assert_equal(first, second)
    before = _rng_state()
    CustomModule.fail = True
    with pytest.raises(RuntimeError, match='custom initialization failed'):
        dict(create_shard_init_weights(CustomModule, {}, seed=17))
    assert_equal(before, _rng_state())
    with pytest.raises(RuntimeError, match='__shard__init__'):
        dict(create_shard_init_weights(torch.nn.Module, {}, seed=17))


@pytest.mark.parametrize('failure', ['shape', 'missing', 'exception'])
def test_streaming_shard_failure_closes_hook_and_restores_rng(failure):
    module = _local_module(ParamInitStrategy.SHARD)
    closed = []

    class CustomModule(torch.nn.Module):
        @staticmethod
        def __shard__init__(attrs):
            try:
                value = torch.rand(2) + random.random() + np.random.rand()
                yield 'weight_local', value
                torch.testing.assert_close(module.weight_local, value)
                if failure == 'shape':
                    yield 'copy_local', torch.ones(3)
                elif failure == 'exception':
                    raise RuntimeError('hook failed')
            finally:
                closed.append(True)

    type(module)._shard_init_fn = staticmethod(partial(create_shard_init_weights, CustomModule, seed=17))
    before = _rng_state()
    with pytest.raises(RuntimeError, match='__shard__init__|hook failed'):
        module._post_init(build_buckets=False)
    assert closed == [True]
    assert_equal(before, _rng_state())


def test_capture_initializer_yields_views_without_cloning():
    """
    Test capture_init_weights yields views without cloning.
    """
    module = _local_module(ParamInitStrategy.SHARD)
    stream = capture_init_weights(module.fullmap, module_fn=type(module)._init_module_fn)
    try:
        name, value = next(stream)
        assert name == 'weight_local'
        assert value.shape == (2,)
        assert value.untyped_storage().nbytes() == 4 * value.element_size()
        assert value.storage_offset() == 1
    finally:
        stream.close()


def test_streaming_user_hook_releases_each_source_before_next_allocation():
    from torch.multiprocessing.reductions import StorageWeakRef

    module = _local_module(ParamInitStrategy.SHARD)
    released = []

    class CustomModule(torch.nn.Module):
        @staticmethod
        def __shard__init__(attrs):
            for index, name in enumerate(attrs):
                full = torch.full((16,), float(index + 1))
                storage = StorageWeakRef(full.untyped_storage())
                yield name, full[1:3]
                del full
                assert storage.expired()
                released.append(name)

    type(module)._shard_init_fn = staticmethod(partial(create_shard_init_weights, CustomModule, seed=17))
    module._post_init(build_buckets=False)
    assert released == ['weight_local', 'copy_local']
    assert torch.equal(module.weight_local, torch.ones(2))
    assert torch.equal(module.copy_local, torch.full((2,), 2.0))


@pytest.mark.parametrize('strategy', [ParamInitStrategy.FULL, ParamInitStrategy.SHARD])
def test_resume_without_nonpersistent_buffers_skips_initializers(strategy):
    module = _local_module(strategy)
    # Simulate the scenario where non-persistent buffers are missing after loading the module.
    del module.copy_local
    del module._fullmap['copy_local']
    with patch.object(type(module), '_init_module_fn', side_effect=AssertionError('constructor')), patch.object(
        type(module), '_shard_init_fn', side_effect=AssertionError('shard constructor'),
    ):
        module._post_init(init_params=False, build_buckets=False)
