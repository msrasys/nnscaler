"""Initialization values and generated tensor placement are separate choices."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import nnscaler
from nnscaler.cli import mixed_module, trainer as trainer_module
from nnscaler.cli.mixed_module import ModuleParallelizeConfigAdapter, parallelize_model
from nnscaler.cli.trainer import Trainer
from nnscaler.parallel import ComputeConfig
from tests.utils import mock_cube_env


def make_adapter(monkeypatch, tmp_path, *, fail=False):
    # Events record (kind, init_params, build_buckets, device) for construction,
    # or (kind, device) for tracing and bucket building.
    calls = []

    class Generated(torch.nn.Module):
        def __init__(self, init_params, build_buckets):
            super().__init__()
            calls.append(('construct', init_params, build_buckets, torch.empty(0).device.type))
            if fail:
                raise RuntimeError('constructor failed')
            self.weight = torch.nn.Parameter(torch.empty(4))
            self.register_buffer('constant', torch.ones(2), persistent=False)

        def build_buckets(self):
            calls.append(('buckets', self.weight.device.type))

    def parallelize(*args, **kwargs):
        calls.append(('trace', torch.empty(0).device.type))
        return Generated

    args = SimpleNamespace(model_type=torch.nn.Linear, gen_savedir=tmp_path,
                           gen_reuse='match', instance_name=None, broadcast_strategy='none',
                           gen_max_workers=1, pas_policy='dp', precision={'grad': 'fp32'})
    adapter = ModuleParallelizeConfigAdapter(args)
    monkeypatch.setattr(adapter, 'resolve_compute_config', lambda: object())
    monkeypatch.setattr(adapter, 'create_dummy_forward_args', lambda _: {})
    monkeypatch.setattr(nnscaler, 'parallelize', parallelize)
    return adapter, calls, Generated


@pytest.mark.parametrize('init_params', [False, True])
@pytest.mark.parametrize('init_device', [None, 'cpu', 'meta', torch.device('cpu')])
def test_explicit_device_scopes_only_generated_construction(monkeypatch, tmp_path, init_params, init_device):
    adapter, calls, _ = make_adapter(monkeypatch, tmp_path)
    result = adapter.parallelize({}, init_params=init_params, init_device=init_device)
    expected = torch.device(init_device or 'cpu').type
    assert result.weight.device.type == expected
    assert result.constant.device.type == expected
    assert calls == [('trace', 'cpu'),
                     ('construct', init_params, False, expected), ('buckets', expected)]
    assert torch.empty(0).device.type == 'cpu'


def test_unspecified_device_preserves_callers_scope(monkeypatch, tmp_path):
    adapter, _, _ = make_adapter(monkeypatch, tmp_path)
    with torch.device('meta'):
        result = adapter.parallelize({}, init_params=False)
    assert result.weight.device.type == 'meta'
    assert torch.empty(0).device.type == 'cpu'


def test_constructor_failure_restores_default_device(monkeypatch, tmp_path):
    adapter, _, _ = make_adapter(monkeypatch, tmp_path, fail=True)
    with pytest.raises(RuntimeError, match='constructor failed'):
        adapter.parallelize({}, init_params=False, init_device='meta')
    assert torch.empty(0).device.type == 'cpu'


def test_compile_only_does_not_construct_or_enter_target_device(monkeypatch, tmp_path):
    adapter, calls, generated = make_adapter(monkeypatch, tmp_path)
    result = adapter.parallelize({}, load_module=False, init_params=False, init_device='cuda:0')
    assert result is generated
    assert calls == [('trace', 'cpu')]


@pytest.mark.parametrize('mixed', [False, True])
def test_whole_and_mixed_paths_forward_explicit_device(monkeypatch, mixed):
    calls = []

    class Child(torch.nn.Module):
        pass

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.outside = torch.nn.Parameter(torch.empty(1))
            self.child = Child()

    def construct(self, *args, **kwargs):
        calls.append(kwargs)
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(ModuleParallelizeConfigAdapter, 'parallelize', construct)
    monkeypatch.setattr(mixed_module, 'fork_rng', nullcontext)
    args = SimpleNamespace(tracing_from_weights=None,
                           model=SimpleNamespace(parallel_modules=[SimpleNamespace(type=Child, tracing_from_weights_prefix=None)] if mixed else []),
                           create_model=Root, to_precision=lambda x: x)
    model = parallelize_model(args, {}, True, False, None, False, init_device='cpu')
    assert len(calls) == 1
    assert calls[0]['init_params'] is False and calls[0]['init_device'] == 'cpu'
    if mixed:
        assert model.outside.device.type == 'cpu'


@pytest.mark.parametrize('resuming', [False, True])
@pytest.mark.parametrize('compile_only', [False, True])
def test_trainer_allocates_generated_parameters_on_cuda(monkeypatch, resuming, compile_only):
    monkeypatch.setattr(trainer_module, 'is_running_distributed', lambda: False)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 2)
    trainer = object.__new__(Trainer)
    trainer.train_args = SimpleNamespace(
        init_env=lambda _: None, create_checkpointer=lambda: None,
        compile_mode=compile_only, dummy_input={},
        checkpoint=SimpleNamespace(get_resume_checkpoint=lambda: 'ckpt' if resuming else None),
        should_delay_bucket_building=lambda: True,
    )
    calls = []

    class ReachedConstruction(Exception):
        pass

    def construct(*args, **kwargs):
        calls.append(kwargs)
        raise ReachedConstruction

    monkeypatch.setattr(trainer_module, 'parallelize_model', construct)
    with pytest.raises(ReachedConstruction):
        trainer._setup()
    assert calls[0]['init_params'] == (not resuming)
    expected = torch.device('cuda', 2) if not compile_only else None
    assert calls[0]['init_device'] == expected


class ModelWithBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.register_buffer('constant', torch.full((4,), 2.0), persistent=False)

    def forward(self, x):
        return x * self.weight + self.constant


@pytest.mark.parametrize('device', [
    'cpu',
    pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu')),
])
@pytest.mark.parametrize('init_params', [False, True])
def test_generated_module_placement_and_non_persistent_buffer(tmp_path, device, init_params):
    args = SimpleNamespace(
        model_type=ModelWithBuffer, create_model=ModelWithBuffer,
        model=SimpleNamespace(args={}), compute_config=ComputeConfig(1, 1, trace_strategy='cpu'),
        gen_savedir=tmp_path, gen_reuse='match', instance_name=None,
        broadcast_strategy='none', gen_max_workers=1, pas_policy='dp',
        precision={'param': 'fp32', 'buffer': 'fp32', 'grad': 'fp32'},
        tracing_from_weights=None, update_freq=1,
        micro_batch_size=1, global_batch_size=1,
    )
    adapter = ModuleParallelizeConfigAdapter(args)
    with mock_cube_env(0, 1):
        torch.distributed.init_process_group(
            'gloo', init_method=(tmp_path / 'store').as_uri(), rank=0, world_size=1,
        )
        try:
            model = adapter.parallelize(
                torch.ones(4), init_params=init_params, build_buckets=False, init_device=device,
            )
        finally:
            torch.distributed.destroy_process_group()
    assert all(t.device == torch.device(device) for t in list(model.parameters()) + list(model.buffers()))
    assert model.non_presistent_buffers_inited
    if init_params:
        torch.testing.assert_close(next(model.parameters()).cpu(), torch.ones(4))
    torch.testing.assert_close(next(model.buffers()).cpu(), torch.full((4,), 2.0))
    assert torch.empty(0).device.type == 'cpu'
