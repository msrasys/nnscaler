# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Order validation must cover multiple partial modules, not just each module."""
from types import SimpleNamespace
import copy
import importlib

import pytest
import torch
import nnscaler

from nnscaler.runtime.module import ParallelModule
from nnscaler.parallel import OptimizerExtraState, _OPTIMIZER_NON_PARALLEL_NAMES_KEY
from tests.parallel_module.test_checkpoint_name_adapter import _ParallelModuleStub
from tests.parallel_module.common import assert_equal


def _modules(reordered):
    old = torch.nn.ModuleDict({'encoder': _ParallelModuleStub(10), 'decoder': _ParallelModuleStub(20)})
    children = [('encoder', _ParallelModuleStub(100)), ('decoder', _ParallelModuleStub(200))]
    new = torch.nn.ModuleDict(dict(reversed(children) if reordered else children))
    for collection in (old, new):
        for module in collection.values():
            module._reducers = []
            module._sync_grad_required = False
            module._non_presistent_buffers_inited = True
            module._register_load_state_dict_pre_hook(ParallelModule._pre_load_state_dict_hook, with_module=True)
    return old, new


def _optimizer(module, wrapped):
    if wrapped:
        return nnscaler.build_optimizer(module, torch.optim.SGD, lr=0.01, momentum=0.9)
    return torch.optim.SGD(module.parameters(), lr=0.01, momentum=0.9)


@pytest.mark.parametrize('wrapped', [False, True])
def test_partial_module_global_optimizer_order_must_be_checked(monkeypatch, wrapped):
    # Only rank lookup is emulated; optimizer construction and checkpoint loading
    # use the real APIs, with CPU tensors and no communication required.
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda group=None: 0)
    parallel = importlib.import_module('nnscaler.parallel')
    monkeypatch.setattr(parallel, 'DeviceGroup', lambda: SimpleNamespace(rank=0, world_size=1))
    old, new = _modules(reordered=True)
    old_optimizer = _optimizer(old, wrapped)
    new_optimizer = _optimizer(new, wrapped)
    for index, param in enumerate(old.parameters(), start=1):
        param.grad = torch.full_like(param, float(index))
    old_optimizer.step()
    with pytest.raises(ValueError, match='[Oo]rder|layout|[Ll]ocation'):
        nnscaler.load_sharded_state_dict(new, old.state_dict(), new_optimizer,
                                        old_optimizer.state_dict(), device='cpu')


@pytest.mark.parametrize('reordered', [False, True])
def test_model_only_name_alignment_ignores_unrelated_key_order(reordered):
    old, new = _modules(reordered)
    state = old.state_dict()
    # Tensor and extra-state key insertion order is not a model identity.
    state = dict(reversed(list(state.items())))
    nnscaler.load_sharded_state_dict(new, state, device='cpu')
    torch.testing.assert_close(new.encoder.weight_100, old.encoder.weight_10)
    torch.testing.assert_close(new.decoder.weight_200, old.decoder.weight_20)


def _rank_zero(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda group=None: 0)
    parallel = importlib.import_module('nnscaler.parallel')
    monkeypatch.setattr(parallel, 'DeviceGroup', lambda: SimpleNamespace(rank=0, world_size=1))


@pytest.mark.parametrize('reordered', [False, True])
@pytest.mark.parametrize('raw', [False, True])
def test_optimizer_layout_is_checked_before_loading_weights(monkeypatch, reordered, raw):
    _rank_zero(monkeypatch)
    old, new = _modules(reordered)
    old_optimizer, new_optimizer = _optimizer(old, not raw), _optimizer(new, not raw)
    old.encoder.weight_10.data.fill_(17.)
    original = new.encoder.weight_100.detach().clone()
    state = old.state_dict()
    optimizer_state = old_optimizer.state_dict()
    if reordered or raw:
        with pytest.raises(ValueError, match='order|layout'):
            nnscaler.load_sharded_state_dict(new, state, new_optimizer, optimizer_state, device='cpu')
        torch.testing.assert_close(new.encoder.weight_100, original)
    else:
        nnscaler.load_sharded_state_dict(new, state, new_optimizer, optimizer_state, device='cpu')
        torch.testing.assert_close(new.encoder.weight_100, old.encoder.weight_10)


def test_wrapped_optimizer_restore_accepts_old_metadata_and_preserves_inputs(monkeypatch):
    _rank_zero(monkeypatch)
    old, new = _modules(False)
    old_optimizer, new_optimizer = _optimizer(old, True), _optimizer(new, True)
    for i, param in enumerate(old.parameters(), 1):
        param.grad = torch.full_like(param, float(i))
    old_optimizer.step()
    state = dict(reversed(list(old.state_dict().items())))
    optimizer_state = old_optimizer.state_dict()
    # A legacy all-parallel optimizer needs no uncompiled parameter names.
    optimizer_state.pop(_OPTIMIZER_NON_PARALLEL_NAMES_KEY, None)
    before = copy.deepcopy(optimizer_state)
    nnscaler.load_sharded_state_dict(new, state, new_optimizer, optimizer_state, device='cpu')
    assert_equal(optimizer_state, before)
    assert_equal(new_optimizer.state_dict()['state'], old_optimizer.state_dict()['state'])


@pytest.mark.parametrize('module_reordered', [False, True])
def test_named_raw_optimizer_follows_actual_parameters_not_module_order(module_reordered):
    old, new = _modules(module_reordered)
    old_named = [(f'{prefix}.{name}', param) for prefix in ('encoder', 'decoder')
                 for name, param in old[prefix].named_parameters()]
    new_named = [(f'{prefix}.{name}', param) for prefix in ('encoder', 'decoder')
                 for name, param in new[prefix].named_parameters()]
    old_optimizer = torch.optim.SGD([{'params': [p for _, p in old_named],
                                     'param_names': [n for n, _ in old_named]}], lr=0.01, momentum=0.9)
    new_optimizer = torch.optim.SGD([p for _, p in new_named], lr=0.01, momentum=0.9)
    for i, (_, param) in enumerate(old_named, 1):
        param.grad = torch.full_like(param, float(i))
    old_optimizer.step()
    state = dict(reversed(list(old.state_dict().items())))
    optimizer_state = old_optimizer.state_dict()
    before = copy.deepcopy(optimizer_state)
    nnscaler.load_sharded_state_dict(new, state, new_optimizer, optimizer_state, device='cpu')
    assert_equal(optimizer_state, before)
    assert new_optimizer.param_groups[0]['param_names'] == [name for name, _ in new_named]
    for (_, old_param), (_, new_param) in zip(old_named, new_named):
        torch.testing.assert_close(new_optimizer.state[new_param]['momentum_buffer'],
                                   old_optimizer.state[old_param]['momentum_buffer'])


def test_named_raw_optimizer_rejects_changed_actual_parameter_order():
    old, new = _modules(True)
    old_named = list(old.named_parameters())
    old_optimizer = torch.optim.SGD([{'params': [p for _, p in old_named],
                                     'param_names': [n for n, _ in old_named]}], lr=0.1)
    new_optimizer = torch.optim.SGD(new.parameters(), lr=0.1)
    with pytest.raises(ValueError, match='global named parameter order'):
        nnscaler.load_sharded_state_dict(new, old.state_dict(), new_optimizer,
                                        old_optimizer.state_dict(), device='cpu')


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('reordered', [False, True])
def test_nonparallel_parameter_order_requires_identity_metadata(monkeypatch, legacy, reordered):
    _rank_zero(monkeypatch)
    old, new = _modules(False)
    old.register_parameter('extra_a', torch.nn.Parameter(torch.ones(2)))
    old.register_parameter('extra_b', torch.nn.Parameter(torch.ones(2)))
    for name in (['extra_b', 'extra_a'] if reordered else ['extra_a', 'extra_b']):
        new.register_parameter(name, torch.nn.Parameter(torch.ones(2)))
    old_optimizer, new_optimizer = _optimizer(old, True), _optimizer(new, True)
    state = old_optimizer.state_dict()
    if legacy:
        state.pop(_OPTIMIZER_NON_PARALLEL_NAMES_KEY)
    if legacy or reordered:
        with pytest.raises(ValueError, match='[Oo]rder'):
            nnscaler.load_sharded_state_dict(new, old.state_dict(), new_optimizer, state, device='cpu')
    else:
        nnscaler.load_sharded_state_dict(new, old.state_dict(), new_optimizer, state, device='cpu')


def test_deduped_layout_rejection_precedes_distributed_loading(monkeypatch):
    _rank_zero(monkeypatch)
    old, new = _modules(True)
    with pytest.raises(ValueError, match='global parameter layout'):
        nnscaler.load_deduped_state_dict(new, old.state_dict(), _optimizer(new, True),
                                        _optimizer(old, True).state_dict(), device='cpu')


@pytest.mark.parametrize('nonparallel', [False, True])
def test_optimizer_identity_does_not_extend_legacy_extra_state_schema(monkeypatch, nonparallel):
    _rank_zero(monkeypatch)
    old, _ = _modules(False)
    if nonparallel:
        old.register_parameter('extra', torch.nn.Parameter(torch.ones(2)))
    state = _optimizer(old, True).state_dict()
    # These are the exact fields accepted by the baseline943 constructor.
    legacy_fields = {'rank', 'name', 'parallel_module_locs', 'parallel_module_configs',
                     'non_parallel_extra_state', 'non_parallel_param_locs'}
    assert set(state[ParallelModule.EXTRA_STATE_KEY]) == legacy_fields
    OptimizerExtraState(**state[ParallelModule.EXTRA_STATE_KEY])
    if nonparallel:
        assert state[_OPTIMIZER_NON_PARALLEL_NAMES_KEY] == ['extra']
    else:
        assert _OPTIMIZER_NON_PARALLEL_NAMES_KEY not in state
