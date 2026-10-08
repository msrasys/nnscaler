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
from nnscaler.runtime.initialization import create_init_module, create_partial_init_weights
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


def test_seeded_initialization_restores_rng_on_failure():
    def failing_factory():
        RandomInitModule('cuda' if torch.cuda.is_available() else 'cpu')
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


@pytest.mark.parametrize('strategy', [ParamInitStrategy.RECREATE, ParamInitStrategy.CAPTURE])
def test_source_strategy_preserves_buffer_dependency_on_init_and_resume(strategy):
    module = _local_module(strategy)
    with patch.object(module, 'load_np_buffer_content', side_effect=AssertionError('buffer file read')):
        module._post_init(build_buckets=False)
        assert torch.equal(module.weight_local, module.copy_local)
        initial_buffer = module.copy_local.clone()
        with torch.no_grad():
            module.weight_local.fill_(9)
            module.copy_local.fill_(0)
        module._post_init(init_params=False, build_buckets=False)
        assert torch.equal(module.weight_local, torch.full((2,), 9.0))
        assert torch.equal(module.copy_local, initial_buffer)
        assert module.non_presistent_buffers_inited


def test_custom_partial_initializer_receives_only_required_attributes():
    module = _local_module(ParamInitStrategy.CUSTOM)
    calls = []

    class CustomModule(torch.nn.Module):
        def __init__(self):
            raise AssertionError('full constructor invoked')

        @staticmethod
        def __partial__init__(attrs):
            calls.append(dict(attrs))
            return {name: torch.full(meta.sub_shape, 7.0, dtype=meta.dtype)
                    for name, meta in attrs.items()}

    type(module)._partial_init_fn = staticmethod(partial(
        create_partial_init_weights, CustomModule, seed=1234,
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
    (None, 'dictionary'),
    ({}, 'missing attributes'),
    ({'weight_local': torch.ones(2), 'copy_local': torch.ones(2), 'extra': torch.ones(2)}, 'unexpected attributes'),
    ({'weight_local': torch.ones(4), 'copy_local': torch.ones(2)}, 'expected shape'),
    ({'weight_local': torch.ones(2, dtype=torch.float64), 'copy_local': torch.ones(2)}, 'dtype'),
    ({'weight_local': torch.ones(2, device='meta'), 'copy_local': torch.ones(2)}, 'Invalid'),
])
def test_custom_partial_initializer_validates_local_results(result, message):
    module = _local_module(ParamInitStrategy.CUSTOM)
    type(module)._partial_init_fn = staticmethod(lambda attrs: result)
    with pytest.raises(RuntimeError, match=message):
        module._post_init(build_buckets=False)


def test_partial_initializer_seed_and_failure_restore_rng():
    class CustomModule(torch.nn.Module):
        fail = False

        @staticmethod
        def __partial__init__(attrs):
            value = torch.rand(4) + random.random() + np.random.rand()
            if CustomModule.fail:
                raise RuntimeError('custom initialization failed')
            return {'value': value}

    before = _rng_state()
    first = create_partial_init_weights(CustomModule, {}, seed=17)
    assert_equal(before, _rng_state())
    torch.rand(11)
    random.random()
    np.random.rand(3)
    second = create_partial_init_weights(CustomModule, {}, seed=17)
    assert_equal(first, second)
    before = _rng_state()
    CustomModule.fail = True
    with pytest.raises(RuntimeError, match='custom initialization failed'):
        create_partial_init_weights(CustomModule, {}, seed=17)
    assert_equal(before, _rng_state())
    with pytest.raises(RuntimeError, match='__partial__init__'):
        create_partial_init_weights(torch.nn.Module, {}, seed=17)


@pytest.mark.parametrize('strategy', [ParamInitStrategy.RECREATE, ParamInitStrategy.CAPTURE, ParamInitStrategy.CUSTOM])
def test_resume_without_nonpersistent_buffers_skips_initializers(strategy):
    module = _local_module(strategy)
    del module.copy_local
    del module._fullmap['copy_local']
    with patch.object(type(module), '_init_module_fn', side_effect=AssertionError('constructor')), patch.object(
        type(module), '_partial_init_fn', side_effect=AssertionError('partial constructor'),
    ):
        module._post_init(init_params=False, build_buckets=False)
