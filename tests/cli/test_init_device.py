"""Initialization values and generated tensor placement are separate choices."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import nnscaler
from nnscaler.cli import mixed_module, trainer as trainer_module
from nnscaler.cli.mixed_module import ModuleParallelizeConfigAdapter, parallelize_model
from nnscaler.cli.trainer import Trainer


def make_adapter(monkeypatch, tmp_path, *, fail=False):
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
        calls.append(('trace', torch.empty(0).device.type, kwargs['allow_missing_init_weights']))
        return Generated

    args = SimpleNamespace(model_type=torch.nn.Linear, gen_savedir=tmp_path,
                           gen_reuse='match', instance_name=None, broadcast_strategy='none',
                           codegen_workers=1, pas_policy='dp', precision={'grad': 'fp32'})
    adapter = ModuleParallelizeConfigAdapter(args)
    monkeypatch.setattr(adapter, 'resolve_compute_config', lambda: object())
    monkeypatch.setattr(adapter, 'create_dummy_forward_args', lambda _: {})
    monkeypatch.setattr(nnscaler, 'parallelize', parallelize)
    return adapter, calls, Generated


@pytest.mark.parametrize('init_params', [False, True])
@pytest.mark.parametrize('init_device', [None, 'cpu', 'meta'])
def test_explicit_device_scopes_only_generated_construction(monkeypatch, tmp_path, init_params, init_device):
    adapter, calls, _ = make_adapter(monkeypatch, tmp_path)
    # The old experiment variable must not override a library caller's device.
    monkeypatch.setenv('NNSCALER_RESUME_INIT_ON_CUDA', '1')
    result = adapter.parallelize({}, init_params=init_params, init_device=init_device)
    expected = init_device or 'cpu'
    assert result.weight.device.type == expected
    assert result.constant.device.type == expected
    assert calls == [('trace', 'cpu', not init_params),
                     ('construct', init_params, False, expected), ('buckets', expected)]
    assert torch.empty(0).device.type == 'cpu'


def test_unspecified_device_preserves_callers_scope(monkeypatch, tmp_path):
    adapter, calls, _ = make_adapter(monkeypatch, tmp_path)
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
    result = adapter.parallelize({}, load_module=False, init_params=False, init_device='meta')
    assert result is generated
    assert calls == [('trace', 'cpu', True)]


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
@pytest.mark.parametrize('override', [None, '0', '1'])
def test_trainer_defaults_resume_to_cuda_with_opt_out(monkeypatch, resuming, compile_only, override):
    if override is None:
        monkeypatch.delenv('NNSCALER_RESUME_INIT_ON_CUDA', raising=False)
    else:
        monkeypatch.setenv('NNSCALER_RESUME_INIT_ON_CUDA', override)
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
    expected = torch.device('cuda', 2) if resuming and not compile_only and override != '0' else None
    assert calls[0]['init_device'] == expected
