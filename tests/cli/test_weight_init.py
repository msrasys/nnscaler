#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import math
import os
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

import nnscaler
from nnscaler import ComputeConfig, ParamInitStrategy
from nnscaler.cli import Trainer, TrainerArgs
from nnscaler.cli.trainer_args import OptionalComputeConfig
from nnscaler.graph.parser import FxModuleParser
from nnscaler.runtime.initialization import DeferredInitialization, create_init_module
from nnscaler.runtime.module import ParallelModule
from tests.launch_torchrun import launch_torchrun


STRATEGIES = (
    ParamInitStrategy.FILE, ParamInitStrategy.FULL, ParamInitStrategy.SHARD,
)


def _assert_tensor_identical(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    actual_bytes = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    expected_bytes = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


class InitModel(torch.nn.Module):
    initial_parameters = {}

    def __init__(self, dim=16, nlayers=16):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(dim, dim, bias=False) for _ in range(nlayers)
        ])
        self.shared = self.layers[0]
        self.register_buffer('scale', torch.full((dim,), random.random() + np.random.rand()))
        self.register_buffer('offset', torch.rand(dim), persistent=False)
        self.register_buffer('scalar', torch.tensor([0.125], dtype=torch.float64))
        self.constant = torch.tensor([0.25])
        # This must be replayed from the same captured weight, not the eager tracing weight.
        self.register_buffer('derived', self.layers[0].weight[0].detach().clone(), persistent=False)
        self.loss_fn = torch.nn.BCELoss()
        type(self).initial_parameters = dict(self.named_parameters(remove_duplicate=False))

    def forward(self, data):
        x = self.shared(data['data'])
        for layer in self.layers:
            x = layer(x)
        x = x * self.scale + self.offset + self.scalar.float() + self.derived
        x = x + self.constant
        return self.loss_fn(torch.sigmoid(x.float()), data['target'].float())


class HookInitModel(InitModel):
    shard_requests = []

    @staticmethod
    def __shard__init__(attr_meta_map):
        HookInitModel.shard_requests.append(dict(attr_meta_map))

        def shards(orig_name, tensor):
            for attr, meta in attr_meta_map.items():
                if meta.orig_name == orig_name:
                    value = tensor[meta.slicers].to(dtype=meta.dtype)
                    assert tuple(value.shape) == meta.sub_shape
                    yield attr, value

        # Reuse one scratch matrix to advance the eager Linear initialization stream.
        # Each yielded view must be copied before the next initialization overwrites it.
        scratch = torch.empty(16, 16)
        for index in range(16):
            torch.nn.init.kaiming_uniform_(scratch, a=math.sqrt(5))
            yield from shards(f'layers.{index}.weight', scratch)
            if index == 0:
                yield from shards('shared.weight', scratch)
                yield from shards('derived', scratch[0])
        scale = random.random() + np.random.rand()
        offset = torch.rand(16)
        yield from shards('scale', torch.full((16,), scale))
        yield from shards('offset', offset)
        yield from shards('scalar', torch.tensor([0.125], dtype=torch.float64))
        yield from shards('constant', torch.tensor([0.25]))


class ClassmethodHookInitModel(HookInitModel):
    @classmethod
    def __shard__init__(cls, attr_meta_map):
        assert cls is ClassmethodHookInitModel
        yield from super().__shard__init__(attr_meta_map)


INIT_CASES = (
    (ParamInitStrategy.FILE, InitModel),
    (ParamInitStrategy.FULL, InitModel),
    (ParamInitStrategy.SHARD, InitModel),
    (ParamInitStrategy.SHARD, HookInitModel),
)
NON_FILE_CASES = tuple(case for case in INIT_CASES if case[0] != ParamInitStrategy.FILE)


class RandomInitModel(InitModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        samplers = [
            lambda w: torch.rand_like(w),
            lambda w: torch.randn_like(w),
            lambda w: torch.randint_like(w, -3, 4),
            lambda w: torch.randint(-3, 4, w.shape).to(w.dtype),
            lambda w: torch.randperm(w.numel()).reshape(w.shape).to(w.dtype),
            lambda w: torch.normal(torch.zeros_like(w), 0.5),
            lambda w: torch.bernoulli(torch.full_like(w, 0.25)),
            lambda w: torch.poisson(torch.full_like(w, 2.)),
            lambda w: torch.multinomial(torch.ones_like(w), w.shape[1], True).to(w.dtype),
        ]
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                sampled = samplers[index % len(samplers)](layer.weight)
                initialized = torch.empty_like(sampled)
                midpoint = initialized.shape[0] // 2
                initialized[:midpoint].copy_(sampled[:midpoint])
                initialized[midpoint:].copy_(sampled[midpoint:])
                layer.weight.copy_(initialized)
                layer.weight.mul_(0.001)


class GenericInitModel(InitModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with torch.no_grad():
            positions = torch.arange(16, dtype=torch.float32) / 16
            selected = torch.nonzero(positions > 0.5)
            self.offset.copy_((positions.sin() + positions.cos()) / selected.sum())
            count = selected.shape[0]
            scale = positions.sum().item() / count
            scales = [value.item() for value in torch.linspace(0.5, 1.0, len(self.layers))]
            for layer, layer_scale in zip(self.layers, scales):
                torch.nn.init.trunc_normal_(layer.weight, std=0.01, a=-0.02, b=0.02)
                layer.weight.copy_(layer.weight.sin() * scale * layer_scale)


def init_dummy_sample(args):
    return {
        'data': torch.ones(args.micro_batch_size, 16),
        'target': torch.full((args.micro_batch_size, 16), 0.5),
    }


def _args(save_dir, strategy, *, checked=False, policy='tp', zero=0,
          model_class=InitModel, seed=1234):
    name = f'{strategy}-{model_class.__name__}-{checked}'
    return [
        '-f', str(Path(__file__).with_name('trainer_args.yaml')),
        '--model.type', f'{__name__}.{model_class.__name__}',
        '--model.args.nlayers', '16',
        '--dummy_sample_gen_fn', f'{__name__}.init_dummy_sample',
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(zero),
        '--compute_config.param_init_strategy', strategy,
        '--debug.param_init_check', str(checked).lower(),
        '--compute_config.param_init_seed', str(seed),
        '--pas_policy', policy,
        '--compute_config.pas_config.pipeline_nstages', '2',
        '--compute_config.pas_config.pipeline_nmicros', '2',
        '--compute_config.pas_config.pipeline_scheduler', '1f1b',
        '--gen_savedir', str(save_dir / 'gen'),
        '--instance_name', name,
        '--checkpoint.save_dir', str(save_dir / f'ckpt-{name}'),
        '--enable_progress_bar', 'false',
        '--max_train_steps', '2',
        '--max_epochs', '2',
        '--dataset.train_args.size', '16',
        '--dataset.val_args.size', '8',
        '--seed', '1234',
    ]


def _reference(strategy, seed=1234, model_class=InitModel):
    if strategy == ParamInitStrategy.SHARD and not hasattr(model_class, '__shard__init__'):
        with DeferredInitialization() as capture:
            model = create_init_module(model_class, model_class, None, seed=seed)
        return {
            name: capture.materialize(value)
            for name, value in list(model.named_parameters(remove_duplicate=False))
            + list(model.named_buffers(remove_duplicate=False)) + [('constant', model.constant)]
        }
    model = create_init_module(model_class, model_class, None, seed=seed)
    return {
        name: value.detach().clone()
        for name, value in list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False)) + [('constant', model.constant)]
    }


def _assert_initialized(model: ParallelModule, expected, *, non_persistent_only=False):
    np_buffers = model.get_non_persistent_buffers()
    assert any(
        meta.orig_name == 'constant'
        for attr_map in model.attr_meta_maps for meta in attr_map.values()
    )
    for attr, meta in model.fullmap.items():
        if non_persistent_only and attr not in np_buffers:
            continue
        value = expected[meta.orig_name][meta.slicers]
        if meta.val_chunks != 1:
            value = value / meta.val_chunks
        _assert_tensor_identical(getattr(model, attr), value)


def _assert_artifacts(model, strategy):
    assert model.compute_config.param_init_strategy == strategy
    file_based = strategy == ParamInitStrategy.FILE
    assert bool(list(model.module_dir.glob('fullmodel.pt*'))) == file_based
    assert (model.module_dir / FxModuleParser.NON_PERSISTENT_BUFFER_FILE).exists() == (
        strategy != ParamInitStrategy.SHARD
    ), (strategy, model.compute_config.param_init_strategy, model.module_dir)


def _worker_parity(save_dir):
    for strategy, model_class in INIT_CASES:
        cli_args = _args(Path(save_dir), strategy, model_class=model_class, seed=97)
        cli_args += ['--init_env_fn', f'{__name__}.skew_initialization_rng']
        expected = _reference(strategy, seed=97, model_class=model_class)
        _assert_tensor_identical(expected['derived'], expected['layers.0.weight'][0])
        HookInitModel.shard_requests = []

        with patch.object(Trainer, '_train'):
            trainer = Trainer(cli_args)
            trainer.run()
        _assert_initialized(trainer.model, expected)
        if model_class is HookInitModel:
            assert trainer.model.fullmap
            assert HookInitModel.shard_requests == [trainer.model.fullmap]
        _assert_artifacts(trainer.model, strategy)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_param_init_parity(tmp_path):
    """Match local parameters and buffers to seeded references through CLI setup.

    Cover FILE, FULL, automatic SHARD and the user shard hook on four ranks with
    different ambient RNG states. Check saved initialization files; skip training.
    """
    launch_torchrun(4, _worker_parity, tmp_path)


def _worker_resume(save_dir):
    for strategy, model_class in INIT_CASES:
        cli_args = _args(Path(save_dir), strategy, model_class=model_class)
        expected = _reference(strategy, model_class=model_class)
        trainer = Trainer([*cli_args, '--max_train_steps', '1'])
        trainer.run()
        persistent = {
            attr: getattr(trainer.model, attr).detach().clone()
            for attr in trainer.model.fullmap if attr not in trainer.model.get_non_persistent_buffers()
        }
        HookInitModel.shard_requests = []
        resumed = Trainer([*cli_args, '--checkpoint.resume_from', 'last'])
        with patch.object(Trainer, '_train'):
            resumed.run()
        for attr, value in persistent.items():
            _assert_tensor_identical(getattr(resumed.model, attr), value)
        _assert_initialized(resumed.model, expected, non_persistent_only=True)
        assert resumed.model.non_presistent_buffers_inited
        if model_class is HookInitModel:
            np_meta_map = {
                attr: meta for attr, meta in resumed.model.fullmap.items()
                if attr in resumed.model.get_non_persistent_buffers()
            }
            assert np_meta_map
            assert HookInitModel.shard_requests == [np_meta_map]


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_param_init_resume(tmp_path):
    """Restore checkpoint parameters and initialize non-persistent buffers on resume.

    Train one step to create a real checkpoint for each initialization strategy,
    then inspect resumed values without further training. The shard hook should
    receive only non-persistent attributes.
    """
    launch_torchrun(4, _worker_resume, tmp_path)


class MismatchedBufferModel(InitModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if torch.distributed.get_rank() == 2:
            with torch.no_grad():
                self.offset.add_(1)


def _worker_full_buffer_mismatch(save_dir):
    trainer = Trainer(_args(Path(save_dir), ParamInitStrategy.FULL, model_class=MismatchedBufferModel))
    with patch.object(Trainer, '_train'):
        trainer.run()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_full_buffer_mismatch(tmp_path):
    """Reject FULL initialization when rank 2 differs from saved non-persistent buffers.

    Replica checking is disabled: this error must come from the initializer's
    saved-buffer validation, not from check_param_init.
    """
    with pytest.raises(ChildFailedError, match='differs from') as exc:
        launch_torchrun(4, _worker_full_buffer_mismatch, tmp_path)
    assert 2 in exc.value.failures


def _worker_source_instance(save_dir):
    nnscaler.init()
    cases = (*NON_FILE_CASES, (ParamInitStrategy.SHARD, ClassmethodHookInitModel),
             (ParamInitStrategy.FULL, HookInitModel))
    for strategy, model_class in cases:
        source = create_init_module(model_class, None, None, seed=1234)
        with torch.no_grad():
            for tensor in source.parameters():
                tensor.add_(0.75)
            for tensor in source.buffers():
                tensor.add_(0.5)
            source.constant.add_(0.25)
        expected = {
            name: value.detach().clone()
            for name, value in list(source.named_parameters(remove_duplicate=False))
            + list(source.named_buffers(remove_duplicate=False)) + [('constant', source.constant)]
        }
        HookInitModel.shard_requests = []
        with patch.object(
            model_class, '__init__', side_effect=AssertionError('recreated supplied instance'),
        ), patch.object(
            DeferredInitialization, 'materialize', side_effect=AssertionError('replayed supplied tensor'),
        ):
            model = nnscaler.parallelize(
                source,
                {'data': {'data': torch.ones(2, 16), 'target': torch.full((2, 16), 0.5)}},
                'tp',
                ComputeConfig(2, 4, param_init_strategy=strategy),
                gen_savedir=Path(save_dir) / f'{strategy}-{model_class.__name__}',
                instance_name=strategy,
                build_module_buckets=False,
            )
        _assert_initialized(model, expected)
        _assert_artifacts(model, strategy)
        assert HookInitModel.shard_requests == []


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_param_init_source_instance(tmp_path):
    """Preserve supplied source tensors when parallelizing with FULL or SHARD.

    Use modified source values to prove they are copied without reconstructing
    the source, replaying initialization, or invoking an attached shard hook.
    This exercises parallelize directly rather than Trainer.
    """
    launch_torchrun(4, _worker_source_instance, tmp_path)


def _worker_selective_initialization(save_dir):
    trainer = Trainer(_args(Path(save_dir), ParamInitStrategy.SHARD, policy='hybrid'))
    materialized = []
    original_materialize = DeferredInitialization.materialize

    def checked_materialize(capture, tensor):
        parameters = InitModel.initial_parameters
        assert parameters and all(param.is_meta for param in parameters.values())
        value = original_materialize(capture, tensor)
        materialized.append(tensor)
        return value

    with patch.object(DeferredInitialization, 'materialize', checked_materialize), patch.object(Trainer, '_train'):
        trainer.run()
    expected_parameters = {
        id(InitModel.initial_parameters[meta.orig_name])
        for meta in trainer.model.fullmap.values() if meta.is_param
    }
    all_parameters = {id(t) for t in InitModel.initial_parameters.values()}
    assert {id(t) for t in materialized} & all_parameters == expected_parameters
    assert len(expected_parameters) < len(all_parameters)
    assert len(materialized) >= len(expected_parameters) + len(trainer.model.get_non_persistent_buffers())


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_selective_initialization(tmp_path):
    """Materialize only the parameters assigned to each pipeline stage in SHARD mode.

    Run hybrid parallel CLI setup on four ranks and compare captured parameter
    identities with the generated model's local metadata, without training.
    """
    launch_torchrun(4, _worker_selective_initialization, tmp_path)


def _worker_random_initialization(save_dir, model_class):
    expected = _reference(ParamInitStrategy.SHARD, model_class=model_class)
    args = _args(Path(save_dir), ParamInitStrategy.SHARD, model_class=model_class)
    with patch.object(Trainer, '_train'):
        trainer = Trainer(args)
        trainer.run()
    _assert_initialized(trainer.model, expected)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
@pytest.mark.parametrize('model_class', [RandomInitModel, GenericInitModel])
def test_cli_random_initialization(tmp_path, model_class):
    """Match automatic SHARD initialization to a seeded deferred reference through CLI.

    Cover random in-place/view writes and data-dependent tensor computations;
    inspect initialized tensors after setup and skip training.
    """
    launch_torchrun(4, _worker_random_initialization, tmp_path, model_class)


class UnsupportedInitModel(InitModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 1:
            with torch.no_grad():
                self.layers[0].weight.sin_()


def _worker_capture_failure(save_dir):
    trainer = Trainer([
        *_args(Path(save_dir), ParamInitStrategy.SHARD),
        '--model.type', f'{__name__}.UnsupportedInitModel',
    ])
    with patch.object(Trainer, '_train'):
        trainer.run()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_capture_failure_propagates(tmp_path):
    """Propagate a rank-1 unsupported initialization operation out of the CLI workers."""
    with pytest.raises(ChildFailedError, match='Deferred initialization failed:') as exc:
        launch_torchrun(4, _worker_capture_failure, tmp_path)
    assert 1 in exc.value.failures


def _worker_shard_contract(save_dir):
    cli_args = _args(Path(save_dir), ParamInitStrategy.SHARD, model_class=HookInitModel)
    Trainer([*cli_args, '--run_mode', 'compile']).run()

    def invalid_hook(meta_map):
        name = next(iter(meta_map))
        yield name, torch.empty(meta_map[name].sub_shape, device='meta')

    # Detailed output validation is covered by the runtime tests; exercise CLI propagation here.
    with patch.object(HookInitModel, '__shard__init__', staticmethod(invalid_hook)), patch.object(
        Trainer, '_train',
    ):
        with pytest.raises(RuntimeError, match='__shard__init__'):
            Trainer(cli_args).run()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_shard_param_init_contract(tmp_path):
    """Reject a shard hook returning a meta tensor when loading compiled CLI code."""
    launch_torchrun(4, _worker_shard_contract, tmp_path)


def _worker_broadcast(save_dir):
    os.environ['LOCAL_WORLD_SIZE'] = '2'
    os.environ['GROUP_RANK'] = str(int(os.environ['RANK']) // 2)
    nnscaler.init()
    rank = torch.distributed.get_rank()
    save_dir = Path(save_dir)
    for strategy, model_class in NON_FILE_CASES:
        cli_args = [
            *_args(save_dir, strategy, model_class=model_class),
            '--gen_savedir', str(save_dir / f'node{rank // 2}' / 'gen'),
            '--broadcast_strategy', 'all',
        ]
        Trainer([*cli_args, '--run_mode', 'compile']).run()
        expected = _reference(strategy, model_class=model_class)
        trainer = Trainer(cli_args)
        with patch.object(Trainer, '_train'):
            trainer.run()
        _assert_initialized(trainer.model, expected)
        _assert_artifacts(trainer.model, strategy)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_param_init_broadcast(tmp_path):
    """Initialize FULL and SHARD models after broadcasting code to node-local directories.

    Simulate two nodes with two ranks each on one host. Verify initialized values
    and required artifacts without fullmodel files or training.
    """
    launch_torchrun(4, _worker_broadcast, tmp_path)


def skew_initialization_rng(trainer):
    count = 7 * (torch.distributed.get_rank() + 1)
    for _ in range(count):
        random.random()
    np.random.rand(count)
    torch.rand(count)
    torch.rand(count, device='cuda')


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_cli_param_init_config(strategy):
    """Resolve each CLI initialization strategy and preserve explicit seeds, including zero."""
    config = ComputeConfig(1, 1)
    assert OptionalComputeConfig(param_init_strategy=strategy).resolve(
        config,
    ).param_init_strategy == strategy
    assert OptionalComputeConfig(param_init_seed=0).resolve(config).param_init_seed == 0
    args = TrainerArgs.from_cli([
        *_args(Path('unused-param-init'), strategy),
        '--compute_config.param_init_seed', '17',
    ])
    assert args.compute_config.param_init_strategy == strategy
    assert args.compute_config.param_init_seed == 17
