# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Checkpoint conversion must produce independently loadable shard metadata."""
import ast
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import nnscaler

from nnscaler.parallel import _OPTIMIZER_NON_PARALLEL_NAMES_KEY
from nnscaler.runtime.hybrid_optimizer import HybridOptConfig, HybridSubOptConfig
from nnscaler.runtime.module import ParallelModule
from tests.parallel_module.common import assert_equal
from tests.parallel_module.test_checkpoint_optimizer_layout import _modules


def _cpu_rank(monkeypatch):
    parallel = importlib.import_module('nnscaler.parallel')
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, 'barrier', lambda **kwargs: None)
    monkeypatch.setattr(torch.distributed, 'gather_object', lambda value, result, **kwargs: result.__setitem__(0, value))
    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', lambda *args, **kwargs: None)
    monkeypatch.setattr(parallel, 'DeviceGroup', lambda: SimpleNamespace(rank=0, world_size=1, get_group=lambda ranks: None))


def _prepare(module):
    for pm in module.modules():
        if isinstance(pm, ParallelModule):
            pm.world_size = 1
            pm.runtime_version = nnscaler.__version__
            pm.module_dir = Path(__file__).parent
            pm.attr_meta_maps = [pm.fullmap]
            pm.__class__ = type('CpuCheckpointModule', (type(pm),), {
                'world_size': 1, 'attr_meta_maps': pm.attr_meta_maps,
            }, skip_init=True)
            pm._param_reducer_map = {}
            pm._zero3_param_metadata = {}
            # These fixtures have no reducers. Avoid the CUDA allocation in
            # build_buckets while retaining real HybridOptimizer construction.
            pm.build_buckets = lambda **kwargs: None


def _optimizer(module, hybrid, *, swapped=False):
    if not hybrid:
        return nnscaler.build_optimizer(module, torch.optim.Adam, lr=0.01)
    return nnscaler.build_optimizer(
        module, nnscaler.HybridOptimizer,
        param_clss_fn=lambda name: (int(('encoder' in name) != swapped), 0),
        config=HybridOptConfig(optimizers=[
            HybridSubOptConfig(type=torch.optim.Adam, options={'lr': 0.01}),
            HybridSubOptConfig(type=torch.optim.AdamW, options={'lr': 0.02}),
        ]),
    )


def _checkpoint_cli_namespace():
    # Load the actual CLI conversion functions without running its main guard
    # or process-global distributed patch. Only Trainer GPU setup is mocked.
    path = Path(nnscaler.__file__).parent / 'cli/checkpoint.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    namespace = {}
    exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def _convert(monkeypatch, writer, model, optimizer, merged):
    if writer == 'broadcast':
        model_state, optimizer_state = nnscaler.trimmed_broadcast_merged_state_dict(
            model, merged['model'], optimizer, merged['optimizer'], device='cpu')
        return {'model': model_state, 'optimizer': optimizer_state}
    namespace = _checkpoint_cli_namespace()
    args = SimpleNamespace(compute_config=SimpleNamespace(runtime_ngpus=1),
                           to_dict=lambda: {'checkpoint': {}}, run_mode='run')
    namespace['Trainer'] = lambda **kwargs: SimpleNamespace(
        model=model, optimizer=optimizer, train_args=args, _setup=lambda: None)
    # Conversion changes process rank environment; keep those changes local to
    # this test via monkeypatch's restoration.
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'GROUP_RANK', 'LOCAL_WORLD_SIZE', 'TORCHELASTIC_RUN_ID'):
        monkeypatch.setenv(name, '0')
    return namespace['_trim_merged_checkpoint'](args, merged, 0)


def test_merged_optimizer_excludes_only_internal_layout_metadata(monkeypatch):
    _cpu_rank(monkeypatch)
    model, _ = _modules(False)
    model.register_parameter('uncompiled', torch.nn.Parameter(torch.ones(2)))
    optimizer = _optimizer(model, False)
    state = optimizer.state_dict()
    state['user_metadata'] = {'epoch': 7}
    state['_nnscaler_user_metadata'] = {'keep': True}
    before = copy.deepcopy(state)
    _, merged = nnscaler.merge_state_dicts([model.state_dict()], [state])
    assert set(merged) == {'state', 'param_groups', 'user_metadata', '_nnscaler_user_metadata'}
    assert merged['user_metadata'] == {'epoch': 7}
    assert merged['_nnscaler_user_metadata'] == {'keep': True}
    assert_equal(state, before)


@pytest.mark.parametrize('writer', ['broadcast', 'cli'])
@pytest.mark.parametrize('hybrid', [False, True])
@pytest.mark.parametrize('nonparallel', [False, True])
def test_converted_checkpoint_restores_with_target_layout(monkeypatch, writer, hybrid, nonparallel):
    _cpu_rank(monkeypatch)
    old, new = _modules(False)
    if nonparallel:
        for model in (old, new):
            model.register_parameter('uncompiled', torch.nn.Parameter(torch.ones(2)))
    for model in (old, new):
        _prepare(model)
    old_optimizer = _optimizer(old, hybrid)
    # A merged checkpoint can be redistributed into a different Hybrid layout.
    new_optimizer = _optimizer(new, hybrid, swapped=True)
    for index, param in enumerate(old.parameters(), 1):
        param.grad = torch.full_like(param, float(index))
    old_optimizer.step()
    model_state, optimizer_state = nnscaler.merge_state_dicts([old.state_dict()], [old_optimizer.state_dict()])
    merged = {'model': model_state, 'optimizer': optimizer_state}
    before = copy.deepcopy(merged)
    converted = _convert(monkeypatch, writer, new, new_optimizer, merged)
    nnscaler.load_sharded_state_dict(new, converted['model'], new_optimizer,
                                    converted['optimizer'], device='cpu')
    assert_equal(converted['optimizer'], new_optimizer.state_dict())
    assert_equal(merged, before)
    if nonparallel:
        assert converted['optimizer'][_OPTIMIZER_NON_PARALLEL_NAMES_KEY] == ['uncompiled']


@pytest.mark.parametrize('change', ['classification', 'param_map', 'reverse_param_map', 'children'])
def test_hybrid_layout_change_is_rejected_before_loading_weights(monkeypatch, change):
    _cpu_rank(monkeypatch)
    old, new = _modules(False)
    for model in (old, new):
        _prepare(model)
    old_optimizer = _optimizer(old, True)
    new_optimizer = _optimizer(new, True, swapped=change == 'classification')
    old.encoder.weight_10.data.fill_(17.)
    before = new.encoder.weight_100.detach().clone()
    saved = old_optimizer.state_dict()
    if change != 'classification':
        saved = copy.deepcopy(saved)
        group = saved['param_groups'][0]
        if change == 'children':
            group['children'][0][0]['params'].reverse()
        else:
            group[change].clear()
    with pytest.raises(ValueError, match='[Hh]ybrid.*layout'):
        nnscaler.load_sharded_state_dict(new, old.state_dict(), new_optimizer, saved, device='cpu')
    assert torch.equal(new.encoder.weight_100, before)
