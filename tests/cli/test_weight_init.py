#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from contextlib import ExitStack
from pathlib import Path
import math
import os
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.multiprocessing.reductions import StorageWeakRef

import nnscaler
from nnscaler import ComputeConfig, ParamInitStrategy
from nnscaler.cli import Trainer, TrainerArgs
from nnscaler.cli.trainer_args import OptionalComputeConfig
from nnscaler.graph.parser import FxModuleParser
from nnscaler.runtime.deferred_initialization import DeferredInitialization
from nnscaler.runtime.initialization import create_init_module
from nnscaler.runtime.module import ParallelModule
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_equal


STRATEGIES = (
    ParamInitStrategy.FILE, ParamInitStrategy.RECREATE,
    ParamInitStrategy.CAPTURE, ParamInitStrategy.CUSTOM,
)
NON_FILE_STRATEGIES = tuple(s for s in STRATEGIES if s != ParamInitStrategy.FILE)


def _assert_tensor_identical(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    actual_bytes = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    expected_bytes = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes)


class InitModel(torch.nn.Module):
    initial_parameters = {}
    constructions = 0
    partial_requests = []

    def __init__(self, dim=16, nlayers=16, mismatch=''):
        super().__init__()
        type(self).constructions += 1
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
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 2:
            with torch.no_grad():
                if mismatch == 'weight':
                    self.layers[0].weight.add_(1)
                elif mismatch == 'buffer':
                    # Invisible after conversion to float32, but not bitwise equal.
                    self.scalar.add_(2 ** -45)
                elif mismatch == 'non_persistent':
                    self.offset.add_(1)
                    self.constant.add_(1)
        type(self).initial_parameters = dict(self.named_parameters(remove_duplicate=False))

    @staticmethod
    def __partial__init__(attr_meta_map):
        InitModel.partial_requests.append(dict(attr_meta_map))
        requested = {meta.orig_name for meta in attr_meta_map.values()}
        tensors = {}
        # Reuse one scratch matrix to advance the eager Linear initialization stream.
        # Only requested tensors (and the derived buffer's dependency) are retained.
        scratch = torch.empty(16, 16)
        for index in range(16):
            torch.nn.init.kaiming_uniform_(scratch, a=math.sqrt(5))
            name = f'layers.{index}.weight'
            if name in requested:
                tensors[name] = scratch.clone()
            if index == 0:
                if 'shared.weight' in requested:
                    tensors['shared.weight'] = scratch.clone()
                if 'derived' in requested:
                    tensors['derived'] = scratch[0].clone()
        scale = random.random() + np.random.rand()
        offset = torch.rand(16)
        if 'scale' in requested:
            tensors['scale'] = torch.full((16,), scale)
        if 'offset' in requested:
            tensors['offset'] = offset
        if 'scalar' in requested:
            tensors['scalar'] = torch.tensor([0.125], dtype=torch.float64)
        if 'constant' in requested:
            tensors['constant'] = torch.tensor([0.25])
        result = {
            attr: tensors[meta.orig_name][meta.slicers].to(dtype=meta.dtype).clone()
            for attr, meta in attr_meta_map.items()
        }
        assert all(tuple(result[attr].shape) == meta.sub_shape for attr, meta in attr_meta_map.items())
        return result

    def forward(self, data):
        x = self.shared(data['data'])
        for layer in self.layers:
            x = layer(x)
        x = x * self.scale + self.offset + self.scalar.float() + self.derived
        x = x + self.constant
        return self.loss_fn(torch.sigmoid(x.float()), data['target'].float())


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


def init_dummy_sample(args):
    return {
        'data': torch.ones(args.micro_batch_size, 16),
        'target': torch.full((args.micro_batch_size, 16), 0.5),
    }


def _args(save_dir, strategy, *, checked=False, policy='tp', zero=0, mismatch=''):
    name = f'{strategy}-{checked}'
    return [
        '-f', str(Path(__file__).with_name('trainer_args.yaml')),
        '--model.type', f'{__name__}.InitModel',
        '--model.args.nlayers', '16',
        '--model.args.mismatch', mismatch,
        '--dummy_sample_gen_fn', f'{__name__}.init_dummy_sample',
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(zero),
        '--compute_config.param_init_strategy', strategy,
        '--debug.param_init_check', str(checked).lower(),
        '--compute_config.param_init_seed', '1234',
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


def _reference(strategy, seed=1234, mismatch=''):
    factory = lambda: InitModel(mismatch=mismatch)
    if strategy == ParamInitStrategy.CAPTURE:
        with DeferredInitialization() as capture:
            model = create_init_module(InitModel, factory, None, seed=seed)
        return {
            name: capture.materialize(value)
            for name, value in list(model.named_parameters(remove_duplicate=False))
            + list(model.named_buffers(remove_duplicate=False)) + [('constant', model.constant)]
        }
    model = create_init_module(InitModel, factory, None, seed=seed)
    return {
        name: value.detach().clone()
        for name, value in list(model.named_parameters(remove_duplicate=False))
        + list(model.named_buffers(remove_duplicate=False)) + [('constant', model.constant)]
    }


def _assert_initialized(model, expected, *, non_persistent_only=False):
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
    file_based = strategy == ParamInitStrategy.FILE
    assert bool(list(model.module_dir.glob('fullmodel.pt*'))) == file_based
    assert (model.module_dir / FxModuleParser.NON_PERSISTENT_BUFFER_FILE).exists() == file_based


def _worker_parity(save_dir, policy, zero):
    save_dir = Path(save_dir)
    results = {}
    cases = [(s, True) for s in STRATEGIES] + [(ParamInitStrategy.CAPTURE, False)]
    for strategy, checked in cases:
        cli_args = _args(save_dir, strategy, checked=checked, policy=policy, zero=zero)
        expected = _reference(strategy)
        _assert_tensor_identical(expected['derived'], expected['layers.0.weight'][0])
        original_build = ParallelModule.build_buckets
        InitModel.partial_requests = []

        def checked_build(model, *args, **kwargs):
            _assert_initialized(model, expected)
            if strategy == ParamInitStrategy.CUSTOM:
                assert InitModel.partial_requests == ([model.fullmap] if model.fullmap else [])
            return original_build(model, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(ParallelModule, 'build_buckets', checked_build))
            if strategy != ParamInitStrategy.FILE:
                stack.enter_context(patch.object(
                    ParallelModule, 'load_attr_content', side_effect=AssertionError('fullmodel read'),
                ))
                stack.enter_context(patch.object(
                    ParallelModule, 'load_np_buffer_content', side_effect=AssertionError('npbuffer read'),
                ))
            trainer = Trainer(cli_args)
            trainer.run()
        assert trainer.train_status.finished_train_steps == 2
        _assert_artifacts(trainer.model, strategy)
        if trainer.rank == 0:
            checkpoint_dir = save_dir / f'ckpt-{strategy}-{checked}' / 'last'
            results[strategy, checked] = Trainer._merge_checkpoint(list(checkpoint_dir.glob('*.ckpt')))
        torch.distributed.barrier()

        before = InitModel.constructions
        InitModel.partial_requests = []
        resumed = Trainer([*cli_args, '--max_train_steps', '3', '--checkpoint.resume_from', 'last'])
        original_initialize = ParallelModule._init_from_module
        resume_initializations = []

        def checked_resume_initialize(model, module, *, init_params=True):
            assert init_params is False
            persistent = {
                attr: getattr(model, attr).detach().clone()
                for attr in model.fullmap if attr not in model.get_non_persistent_buffers()
            }
            original_initialize(model, module, init_params=init_params)
            for attr, value in persistent.items():
                _assert_tensor_identical(getattr(model, attr), value)
            resume_initializations.append(model)

        def checked_resume_build(model, *args, **kwargs):
            _assert_initialized(model, expected, non_persistent_only=True)
            if strategy == ParamInitStrategy.CUSTOM:
                np_meta_map = {
                    attr: meta for attr, meta in model.fullmap.items()
                    if attr in model.get_non_persistent_buffers()
                }
                assert InitModel.partial_requests == ([np_meta_map] if np_meta_map else [])
                assert all(not meta.is_param for meta in np_meta_map.values())
            return original_build(model, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(ParallelModule, 'build_buckets', checked_resume_build))
            stack.enter_context(patch.object(ParallelModule, '_init_from_module', checked_resume_initialize))
            stack.enter_context(patch.object(
                ParallelModule, 'load_attr_content', side_effect=AssertionError('resume read fullmodel'),
            ))
            stack.enter_context(patch(
                'nnscaler.cli.trainer.check_param_init',
                side_effect=AssertionError('resume checked checkpoint initialization'),
            ))
            if strategy in (ParamInitStrategy.FILE, ParamInitStrategy.CUSTOM):
                stack.enter_context(patch.object(
                    InitModel, '__init__', side_effect=AssertionError('unexpected source construction'),
                ))
            if strategy != ParamInitStrategy.FILE:
                stack.enter_context(patch.object(
                    ParallelModule, 'load_np_buffer_content', side_effect=AssertionError('resume read npbuffer'),
                ))
            resumed.run()
        assert InitModel.constructions == before + (
            strategy in (ParamInitStrategy.RECREATE, ParamInitStrategy.CAPTURE)
            and bool(resumed.model.get_non_persistent_buffers())
        )
        assert resumed.train_status.finished_train_steps == 3
        assert resumed.model.non_presistent_buffers_inited
        assert bool(resume_initializations) == (strategy != ParamInitStrategy.FILE)
        _assert_artifacts(resumed.model, strategy)

    if torch.distributed.get_rank() == 0:
        for key in ('model', 'optimizer'):
            assert_equal(results[ParamInitStrategy.FILE, True][key],
                         results[ParamInitStrategy.RECREATE, True][key])
            assert_equal(results[ParamInitStrategy.RECREATE, True][key],
                         results[ParamInitStrategy.CUSTOM, True][key])
            assert_equal(results[ParamInitStrategy.CAPTURE, True][key],
                         results[ParamInitStrategy.CAPTURE, False][key])


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
@pytest.mark.parametrize('policy,zero', [('tp', 0), ('tp', 1), ('tp', 3), ('hybrid', 0)])
def test_cli_param_init_parity(tmp_path, policy, zero):
    launch_torchrun(4, _worker_parity, tmp_path, policy, zero)


def _worker_mismatch(save_dir, strategy, mismatch):
    original_partial = InitModel.__partial__init__

    def mismatched_partial(meta_map):
        values = original_partial(meta_map)
        if torch.distributed.get_rank() == 2:
            for name, meta in meta_map.items():
                if mismatch == 'weight' and meta.is_param:
                    values[name].add_(1)
                elif mismatch == 'buffer' and meta.orig_name == 'scalar':
                    values[name].add_(2 ** -45)
                elif mismatch == 'non_persistent' and meta.orig_name in ('offset', 'constant'):
                    values[name].add_(1)
        return values

    with patch.object(InitModel, '__partial__init__', staticmethod(mismatched_partial)):
        trainer = Trainer(_args(Path(save_dir), strategy, checked=True, mismatch=mismatch))
        with pytest.raises(RuntimeError, match='initialization differs across ranks'):
            trainer.run()
        # Rank 2 is a replica of rank 0, not simply a different TP shard.
        torch.distributed.barrier()
        unchecked = Trainer(_args(Path(save_dir), strategy, mismatch=mismatch))
        with patch('nnscaler.cli.trainer.check_param_init',
                   side_effect=AssertionError('disabled param_init_check compared weights')):
            unchecked.run()
    assert unchecked.train_status.finished_train_steps == 2


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
@pytest.mark.parametrize('strategy', ['recreate', 'capture', 'custom'])
@pytest.mark.parametrize('mismatch', ['weight', 'buffer', 'non_persistent'])
def test_cli_param_init_mismatch(tmp_path, strategy, mismatch):
    launch_torchrun(4, _worker_mismatch, tmp_path, strategy, mismatch)


def _worker_source_buffers(save_dir):
    nnscaler.init()
    for strategy in (ParamInitStrategy.RECREATE, ParamInitStrategy.CAPTURE):
        # Independently random NP buffers follow this strategy's stream, while derived
        # buffers must agree with its own weight (never an eager/captured mixture).
        expected = _reference(strategy, mismatch='non_persistent')
        trainer = Trainer(_args(Path(save_dir), strategy, mismatch='non_persistent'))
        original_build = ParallelModule.build_buckets

        def checked_build(model, *args, **kwargs):
            _assert_initialized(model, expected)
            _assert_artifacts(model, strategy)
            _assert_tensor_identical(expected['derived'], expected['layers.0.weight'][0])
            return original_build(model, *args, **kwargs)

        with patch.object(ParallelModule, 'build_buckets', checked_build):
            trainer.run()
        assert trainer.train_status.finished_train_steps == 2


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_param_init_source_buffers(tmp_path):
    launch_torchrun(4, _worker_source_buffers, tmp_path)


def _worker_source_instance(save_dir):
    nnscaler.init()
    for strategy in (ParamInitStrategy.RECREATE, ParamInitStrategy.CAPTURE):
        source = create_init_module(InitModel, None, None, seed=1234)
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
        with patch.object(
            InitModel, '__init__', side_effect=AssertionError('recreated supplied instance'),
        ), patch.object(
            DeferredInitialization, 'materialize', side_effect=AssertionError('replayed supplied tensor'),
        ):
            model = nnscaler.parallelize(
                source,
                {'data': {'data': torch.ones(2, 16), 'target': torch.full((2, 16), 0.5)}},
                'tp',
                ComputeConfig(2, 4, param_init_strategy=strategy),
                gen_savedir=Path(save_dir) / strategy,
                build_module_buckets=False,
            )
        _assert_initialized(model, expected)
        _assert_artifacts(model, strategy)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_param_init_source_instance(tmp_path):
    launch_torchrun(4, _worker_source_instance, tmp_path)


def _worker_selective_initialization(save_dir):
    trainer = Trainer(_args(Path(save_dir), ParamInitStrategy.CAPTURE, policy='hybrid', checked=True))
    materialized = []
    storages = []
    original_materialize = DeferredInitialization.materialize

    def checked_materialize(capture, tensor):
        assert all(storage.expired() for storage in storages)
        parameters = InitModel.initial_parameters
        assert parameters and all(param.is_meta for param in parameters.values())
        value = original_materialize(capture, tensor)
        materialized.append(tensor)
        storages.append(StorageWeakRef(value.untyped_storage()))
        return value

    with patch.object(DeferredInitialization, 'materialize', checked_materialize):
        trainer.run()
    expected_parameters = {
        id(InitModel.initial_parameters[meta.orig_name])
        for meta in trainer.model.fullmap.values() if meta.is_param
    }
    all_parameters = {id(t) for t in InitModel.initial_parameters.values()}
    assert {id(t) for t in materialized} & all_parameters == expected_parameters
    assert len(expected_parameters) < len(all_parameters)
    assert len(materialized) >= len(expected_parameters) + len(trainer.model.get_non_persistent_buffers())
    assert all(storage.expired() for storage in storages)
    assert trainer.train_status.finished_train_steps == 2


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_selective_initialization(tmp_path):
    launch_torchrun(4, _worker_selective_initialization, tmp_path)


def _worker_random_initialization(save_dir):
    with DeferredInitialization() as capture:
        source = create_init_module(RandomInitModel, None, None, seed=1234)
    expected = {
        name: capture.materialize(value)
        for name, value in list(source.named_parameters(remove_duplicate=False))
        + list(source.named_buffers(remove_duplicate=False)) + [('constant', source.constant)]
    }
    args = _args(Path(save_dir), ParamInitStrategy.CAPTURE, checked=True)
    args += ['--model.type', f'{__name__}.RandomInitModel']
    original_build = ParallelModule.build_buckets

    def checked_build(model, *args, **kwargs):
        _assert_initialized(model, expected)
        return original_build(model, *args, **kwargs)

    with patch.object(ParallelModule, 'build_buckets', checked_build):
        trainer = Trainer(args)
        trainer.run()
    assert trainer.train_status.finished_train_steps == 2
    _assert_artifacts(trainer.model, ParamInitStrategy.CAPTURE)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_random_initialization(tmp_path):
    launch_torchrun(4, _worker_random_initialization, tmp_path)


class UnsupportedInitModel(InitModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 1:
            self.layers[0].weight.sum().item()


def _worker_capture_failure(save_dir):
    trainer = Trainer([
        *_args(Path(save_dir), ParamInitStrategy.CAPTURE, checked=True),
        '--model.type', f'{__name__}.UnsupportedInitModel',
    ])
    trainer.run()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_capture_failure_propagates(tmp_path):
    with pytest.raises(ChildFailedError, match='Deferred initialization failed:') as exc:
        launch_torchrun(4, _worker_capture_failure, tmp_path)
    assert 1 in exc.value.failures


def _worker_custom_contract(save_dir):
    cli_args = _args(Path(save_dir), ParamInitStrategy.CUSTOM, checked=True)
    Trainer([*cli_args, '--run_mode', 'compile']).run()
    valid_hook = InitModel.__partial__init__

    def invalid_hook(meta_map, case):
        result = valid_hook(meta_map)
        name = next(iter(result))
        if case == 'result':
            return list(result.values())
        if case == 'missing':
            result.pop(name)
        elif case == 'extra':
            result['unknown_generated_attr'] = torch.zeros(1)
        elif case == 'type':
            result[name] = 'not a tensor'
        elif case == 'shape':
            result[name] = torch.empty(result[name].numel() + 1)
        elif case == 'dtype':
            result[name] = result[name].to(torch.int64)
        elif case == 'meta':
            result[name] = torch.empty_like(result[name], device='meta')
        return result

    for case in ('absent', 'hook', 'non_static', 'result', 'missing', 'extra', 'type', 'shape', 'dtype', 'meta'):
        trainer = Trainer(cli_args)
        hook = (
            None if case == 'hook' else
            (lambda self, meta_map: valid_hook(meta_map)) if case == 'non_static' else
            staticmethod(lambda meta_map: invalid_hook(meta_map, case))
        )
        with patch.object(InitModel, '__partial__init__', hook), patch.object(
            InitModel, '__init__', side_effect=AssertionError('custom constructed source'),
        ):
            if case == 'absent':
                del InitModel.__partial__init__
            with pytest.raises((RuntimeError, ValueError, TypeError), match='partial|initialization'):
                trainer.run()
        torch.distributed.barrier()
    trainer = Trainer(cli_args)
    with patch.object(InitModel, '__init__', side_effect=AssertionError('custom constructed source')):
        trainer.run()
    _assert_artifacts(trainer.model, ParamInitStrategy.CUSTOM)
    assert trainer.train_status.finished_train_steps == 2


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_custom_param_init_contract(tmp_path):
    launch_torchrun(4, _worker_custom_contract, tmp_path)


def _worker_broadcast(save_dir):
    os.environ['LOCAL_WORLD_SIZE'] = '2'
    os.environ['GROUP_RANK'] = str(int(os.environ['RANK']) // 2)
    nnscaler.init()
    rank = torch.distributed.get_rank()
    save_dir = Path(save_dir)
    for strategy in NON_FILE_STRATEGIES:
        cli_args = [
            *_args(save_dir, strategy, checked=True),
            '--gen_savedir', str(save_dir / f'node{rank // 2}' / 'gen'),
            '--precision', 'bf16',
        ]
        Trainer([*cli_args, '--run_mode', 'compile']).run()
        before = InitModel.constructions
        trainer = Trainer(cli_args)
        with patch.object(ParallelModule, 'load_attr_content', side_effect=AssertionError('fullmodel read')):
            trainer.run()
        assert InitModel.constructions == before + (
            strategy != ParamInitStrategy.CUSTOM and bool(trainer.model.fullmap)
        )
        _assert_artifacts(trainer.model, strategy)
        assert all(p.dtype == torch.bfloat16 for p in trainer.model.parameters())
        assert trainer.train_status.finished_train_steps == 2


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_param_init_broadcast(tmp_path):
    launch_torchrun(4, _worker_broadcast, tmp_path)


def _worker_mixed(save_dir):
    save_dir = Path(save_dir)
    for strategy in (ParamInitStrategy.RECREATE, ParamInitStrategy.CAPTURE):
        trainer = Trainer([
            '-f', str(Path(__file__).with_name('trainer_args_mixed1.yaml')),
            '--compute_config.param_init_strategy', strategy,
            '--debug.param_init_check', 'true',
            '--instance_name', f'mixed-{strategy}',
            '--gen_savedir', str(save_dir / 'gen'),
            '--checkpoint.save_dir', str(save_dir / f'ckpt-{strategy}'),
            '--max_train_steps', '2',
            '--enable_progress_bar', 'false',
        ])
        with patch.object(ParallelModule, 'load_attr_content', side_effect=AssertionError('fullmodel read')):
            trainer.run()
        modules = [m for m in trainer.model.modules() if isinstance(m, ParallelModule)]
        assert len(modules) == 1
        assert modules[0].compute_config.param_init_strategy == strategy
        _assert_artifacts(modules[0], strategy)
        assert trainer.train_status.finished_train_steps == 2


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_param_init_mixed(tmp_path):
    launch_torchrun(4, _worker_mixed, tmp_path)


def skew_initialization_rng(trainer):
    count = 7 * (torch.distributed.get_rank() + 1)
    for _ in range(count):
        random.random()
    np.random.rand(count)
    torch.rand(count)
    torch.rand(count, device='cuda')


def _worker_init_seed(save_dir):
    nnscaler.init()
    for index, strategy in enumerate(NON_FILE_STRATEGIES):
        seed = 97 + index
        expected = _reference(strategy, seed)
        trainer = Trainer([
            *_args(Path(save_dir), strategy, checked=True),
            '--compute_config.param_init_seed', str(seed),
            '--init_env_fn', f'{__name__}.skew_initialization_rng',
        ])
        original_build = ParallelModule.build_buckets

        def checked_build(model, *args, **kwargs):
            _assert_initialized(model, expected)
            return original_build(model, *args, **kwargs)

        with patch.object(ParallelModule, 'build_buckets', checked_build):
            trainer.run()
        assert trainer.train_status.finished_train_steps == 2


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_param_init_seed(tmp_path):
    launch_torchrun(4, _worker_init_seed, tmp_path)


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_cli_param_init_config(strategy):
    config = ComputeConfig(1, 1)
    assert OptionalComputeConfig(param_init_strategy=strategy).resolve(
        config,
    ).param_init_strategy == strategy
    assert OptionalComputeConfig(param_init_seed=0).resolve(config).param_init_seed == 0
    args = TrainerArgs.from_cli([
        *_args(Path('unused-param-init'), strategy, checked=True),
        '--compute_config.param_init_seed', '17',
    ])
    assert args.compute_config.param_init_strategy == strategy
    assert args.compute_config.param_init_seed == 17
    assert args.debug.param_init_check is True
