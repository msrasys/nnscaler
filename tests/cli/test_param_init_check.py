#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nnscaler import ParamInitStrategy
from nnscaler.cli import Trainer, TrainerArgs
from nnscaler.cli.trainer import check_param_init
from nnscaler.cli.trainer_args import DebugConfig, ModelConfig, ModuleParallelizeConfig
from nnscaler.runtime.module import AttrMeta, ParallelModule, Zero3AttrMeta
from tests.cli.test_weight_init import _args
from tests.launch_torchrun import launch_torchrun


class CheckBlock(torch.nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, input):
        return self.linear(input)


class MixedCheckModel(torch.nn.Module):
    mismatch = ''

    def __init__(self):
        super().__init__()
        self.first = CheckBlock()
        self.second = CheckBlock()
        self.ordinary_weight = torch.nn.Parameter(torch.ones(16))
        self.register_buffer('ordinary_buffer', torch.full((16,), 0.125, dtype=torch.float64))
        self.register_buffer('ordinary_nonpersistent', torch.full((16,), 0.25), persistent=False)
        with torch.no_grad():
            # Identical generated layouts at different paths may have different valid values.
            for parameter in self.first.parameters():
                parameter.fill_(0.125)
            for parameter in self.second.parameters():
                parameter.fill_(0.25)
            if torch.distributed.get_rank() == 2:
                mismatch = type(self).mismatch
                if mismatch == 'parameter':
                    self.ordinary_weight.add_(1)
                elif mismatch == 'parallel':
                    next(self.first.parameters()).add_(1)

    def forward(self, data):
        x = self.second(self.first(data['data'])) * self.ordinary_weight
        x = x + self.ordinary_buffer.float() + self.ordinary_nonpersistent
        return torch.nn.functional.binary_cross_entropy(torch.sigmoid(x), data['target'])


def block_forward_args(args):
    return {'input': torch.ones(args.micro_batch_size, 16)}


def _mixed_args(save_dir, strategy, checked=True):
    args = TrainerArgs.from_cli(_args(Path(save_dir), strategy, checked=checked))
    args.model = ModelConfig(
        type=f'{__name__}.MixedCheckModel',
        parallel_modules=[
            ModuleParallelizeConfig(
                type=f'{__name__}.CheckBlock',
                args={'dim': 16},
                forward_args_gen_fn=block_forward_args,
            ),
        ],
    )
    args.max_train_steps = 1
    return args


def _worker_whole_model(save_dir):
    for mismatch in ('', 'parameter', 'parallel'):
        trainer = Trainer(train_args=_mixed_args(save_dir, ParamInitStrategy.FULL))
        # Inject runtime differences without changing the generated-code cache key.
        with patch.object(MixedCheckModel, 'mismatch', mismatch), patch.object(
            Trainer, '_train',
        ), patch('nnscaler.cli.trainer.check_param_init', wraps=check_param_init) as check:
            if mismatch:
                with pytest.raises(RuntimeError, match='initialization differs across ranks') as exc:
                    trainer.run()
                assert 'ranks 0, 2' in str(exc.value)
                assert ('first' if mismatch == 'parallel' else 'ordinary_weight') in str(exc.value)
            else:
                trainer.run()
            check.assert_called_once_with(trainer.model)
        torch.distributed.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_param_init_check_whole_model(tmp_path):
    """Run the CLI check on four ranks with ordinary and parallel parameters.

    Accept matching replicas at distinct submodule paths, then reject a rank-2
    mismatch in either an ordinary parameter or a parallel parameter.
    """
    launch_torchrun(4, _worker_whole_model, tmp_path)


class ZeroCheckModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = CheckBlock()
        self.ordinary_weight = torch.nn.Parameter(torch.ones(16))

    def forward(self, data):
        value = torch.sigmoid(self.block(data['data']) * self.ordinary_weight)
        return torch.nn.functional.binary_cross_entropy(value, data['target'])


def _worker_post_zero3(save_dir):
    args = _mixed_args(save_dir, ParamInitStrategy.FULL)
    args.compute_config = replace(
        args.compute_config, plan_ngpus=1, use_zero=3, zero_ngroups=2,
    )
    args.model.type = f'{__name__}.ZeroCheckModel'

    def checked_init(model):
        assert trainer.optimizer is not None
        assert isinstance(model.block, ParallelModule)
        attr, = model.block.fullmap
        metadata = model.block.get_zero3_attr_meta(attr)
        assert metadata is not None
        assert 0 < metadata.end - metadata.start < 16 * 16
        if inject_mismatch and torch.distributed.get_rank() == 2:
            with torch.no_grad():
                getattr(model.block, attr)[0].add_(1)
        check_param_init(model)

    for inject_mismatch in (False, True):
        trainer = Trainer(train_args=args)
        with patch.object(Trainer, '_train'), patch(
            'nnscaler.cli.trainer.check_param_init', side_effect=checked_init,
        ) as check:
            if inject_mismatch:
                with pytest.raises(RuntimeError, match='initialization differs across ranks') as exc:
                    trainer.run()
                assert "'block'" in str(exc.value)
                assert 'ranks 0, 2' in str(exc.value)
            else:
                trainer.run()
            check.assert_called_once_with(trainer.model)
        torch.distributed.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires four GPUs')
def test_cli_param_init_check_post_zero3(tmp_path):
    """Run the CLI check after ZeRO-3 sharding across two groups on four ranks.

    Accept matching retained shards, then reject a rank-2 modification to a
    shard replicated on rank 0.
    """
    launch_torchrun(4, _worker_post_zero3, tmp_path)


def test_param_init_check_default():
    """Enable initialization replica checking by default in the CLI debug config."""
    assert DebugConfig().param_init_check is True


@pytest.mark.parametrize('initialized', [False, True])
def test_check_without_peers_skips_hashing(initialized):
    """Skip both hashing and communication when distributed is absent or has one rank."""
    with patch('nnscaler.cli.trainer.dist.is_initialized', return_value=initialized), patch(
        'nnscaler.cli.trainer.dist.get_world_size', return_value=1,
    ), patch('nnscaler.cli.trainer._initialization_digest') as digest, patch(
        'nnscaler.cli.trainer.dist.all_gather_object',
    ) as gather:
        check_param_init(_plain_model())
        digest.assert_not_called()
        gather.assert_not_called()


def _plain_model(value=0.125):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2))
    model.register_buffer('persistent', torch.tensor([value], dtype=torch.float64))
    model.register_buffer('nonpersistent', torch.ones(2), persistent=False)
    return model


def _payload(model):
    with patch('nnscaler.cli.trainer.dist.is_initialized', return_value=True), patch(
        'nnscaler.cli.trainer.dist.get_world_size', return_value=4,
    ), patch('nnscaler.cli.trainer.dist.all_gather_object') as gather:
        gather.side_effect = lambda outputs, value: outputs.__setitem__(slice(None), [value] * 4)
        check_param_init(model)
        gather.assert_called_once()
        return gather.call_args.args[1]


def _check_payloads(model, payloads, match=None):
    with patch('nnscaler.cli.trainer.dist.is_initialized', return_value=True), patch(
        'nnscaler.cli.trainer.dist.get_world_size', return_value=4,
    ), patch('nnscaler.cli.trainer.dist.all_gather_object') as gather:
        gather.side_effect = lambda outputs, value: outputs.__setitem__(slice(None), payloads)
        if match is None:
            check_param_init(model)
        else:
            with pytest.raises(RuntimeError, match=match):
                check_param_init(model)
        gather.assert_called_once()


@pytest.mark.parametrize('kind', ['weight', 'persistent', 'nonpersistent'])
def test_check_plain_model_replica(kind):
    """With mocked ranks, accept equal ordinary tensors and reject a changed replica.

    Cover parameters, persistent buffers (including tiny float64 changes), and
    non-persistent buffers.
    """
    model = _plain_model()
    changed = _plain_model()
    with torch.no_grad():
        getattr(changed, kind).add_(2 ** -45 if kind == 'persistent' else 1)
    reference = _payload(model)
    assert len(reference[0]) == 3
    _check_payloads(model, [reference] * 4)
    _check_payloads(model, [reference, reference, _payload(changed), reference],
                    'initialization differs across ranks')


@pytest.mark.parametrize('difference', ['signed_zero', 'shape', 'dtype'])
def test_check_tensor_representation(difference):
    """Reject mocked buffer replicas differing in signed zero, shape, or dtype."""
    value = torch.zeros(2)
    if difference == 'signed_zero':
        changed = -value
    elif difference == 'shape':
        changed = value.reshape(1, 2)
    else:
        changed = value.double()
    model = torch.nn.Module()
    model.register_buffer('value', value)
    replica = torch.nn.Module()
    replica.register_buffer('value', changed)
    reference = _payload(model)
    _check_payloads(model, [reference, reference, _payload(replica), reference],
                    'initialization differs across ranks')


def _parallel_stub(value, *, strategy=ParamInitStrategy.FULL, shape=(4,), start=0, chunks=1):
    class LocalModule(ParallelModule, skip_init=True):
        pass

    model = object.__new__(LocalModule)
    torch.nn.Module.__init__(model)
    model.compute_config = SimpleNamespace(param_init_strategy=strategy)
    model._zero3_param_metadata = {}
    model.weight = torch.nn.Parameter(torch.full((2,), value))
    model._fullmap = {
        'weight': AttrMeta(
            tid=0, is_param=True, orig_name='weight', shape=shape,
            slicers=(slice(start, start + 2),), val_chunks=chunks,
            dtype=torch.float32, sub_shape=(2,),
        ),
    }
    return model


@pytest.mark.parametrize('layout', [{'start': 2}, {'shape': (8,)}, {'chunks': 2}])
def test_check_distinct_logical_shards(layout):
    """Compare only matching logical shards in mocked collective payloads.

    Different slice offsets, full shapes, or value-partition factors may have
    different values; a changed replica of the same logical shard must fail.
    """
    model = _parallel_stub(0.125)
    reference = _payload(model)
    other = _payload(_parallel_stub(0.25, **layout))
    _check_payloads(model, [reference, other, reference, other])
    changed_replica = _payload(_parallel_stub(0.5))
    _check_payloads(model, [reference, other, changed_replica, other],
                    'initialization differs across ranks')


def _zero3_stub(values, start, end, ranks=(0, 1), *, numel=8):
    model = _parallel_stub(0.0, shape=(numel,))
    model.weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
    model._fullmap['weight'].slicers = (slice(0, numel),)
    model._fullmap['weight'].sub_shape = (numel,)
    model._zero3_param_metadata['weight'] = Zero3AttrMeta(
        orig_name='weight', attr_name='weight', start=start, end=end, chunk_size=len(values),
    )
    model._reducers = [SimpleNamespace(ranks=list(ranks))]
    return model


def test_check_zero3_range_and_cross_group_replicas():
    """Compare identical retained ZeRO-3 intervals across mocked reducer groups.

    Distinct intervals may differ, but a changed replica of the same interval
    must fail even when its reducer belongs to another group.
    """
    model = _zero3_stub([1, 2], 0, 2)
    reference = _payload(model)
    other = _payload(_zero3_stub([3, 4], 2, 4))
    replica = _payload(_zero3_stub([1, 2], 0, 2, ranks=(2, 3)))
    assert reference == replica
    assert next(iter(reference[0]))[-1] == (0, 2)
    assert next(iter(other[0]))[-1] == (2, 4)
    _check_payloads(model, [reference, other, replica, other])
    mismatch = _payload(_zero3_stub([9, 2], 0, 2, ranks=(2, 3)))
    _check_payloads(model, [reference, other, mismatch, other],
                    'initialization differs across ranks')


def test_check_zero3_unique_ranges():
    """Accept distinct values when every mocked rank retains a unique ZeRO-3 interval."""
    models = [_zero3_stub([start + 1, start + 2], start, start + 2) for start in range(0, 8, 2)]
    _check_payloads(models[0], [_payload(model) for model in models])


def test_check_zero3_valid_tail_ignores_padding():
    """Ignore ZeRO-3 padding values and lengths, but reject changes in the valid tail."""
    model = _zero3_stub([5, 6, 7, 123], 4, 7, numel=7)
    reference = _payload(model)
    replica = _payload(_zero3_stub([5, 6, 7, -123, 456], 4, 7, numel=7))
    assert reference == replica
    _check_payloads(model, [reference, replica, reference, replica])
    mismatch = _payload(_zero3_stub([5, 6, 8, 123], 4, 7, numel=7))
    _check_payloads(model, [reference, replica, mismatch, replica],
                    'initialization differs across ranks')


@pytest.mark.parametrize('start,end', [(8, 8), (8, 7)])
def test_check_zero3_empty_ranges(start, end):
    """Exclude ZeRO-3 shards with empty or reversed retained intervals from hashing."""
    assert _payload(_zero3_stub([123, 456], start, end))[0] == {}


def test_check_frozen_parallel_parameter_replica():
    """
    Hash a frozen parallel parameter and reject a changed mocked replica.
    """
    model = _parallel_stub(0.125)
    model.weight.requires_grad_(False)
    reference = _payload(model)
    assert next(iter(reference[0]))[-1] is None
    with torch.no_grad():
        model.weight.add_(1)
    _check_payloads(model, [reference, reference, _payload(model), reference],
                    'initialization differs across ranks')


def test_check_file_module_excludes_only_parallel_tensors():
    """Skip FILE-backed parallel tensors without skipping ordinary tensors in the model."""
    model = _plain_model()
    model.parallel = _parallel_stub(0.125, strategy=ParamInitStrategy.FILE)
    reference = _payload(model)
    with torch.no_grad():
        model.parallel.weight.add_(1)
    assert _payload(model) == reference
    with torch.no_grad():
        model.weight.add_(1)
    _check_payloads(model, [reference, reference, _payload(model), reference],
                    'initialization differs across ranks')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='dummy input requires CUDA')
@pytest.mark.parametrize('checked', [False, True])
def test_parallelize_failure_skips_check(tmp_path, checked):
    """Propagate a parallelize_model failure without checking an unconstructed model.

    Cover both enabled and disabled checking; CUDA is needed to load the dummy
    input before reaching the mocked construction failure.
    """
    trainer = Trainer(train_args=_mixed_args(tmp_path, ParamInitStrategy.FULL, checked))
    error = ValueError('source construction failed')
    with patch('nnscaler.cli.trainer.is_running_distributed', return_value=False), patch.object(
        trainer.train_args, 'init_env',
    ), patch('nnscaler.cli.trainer.parallelize_model', side_effect=error), patch(
        'nnscaler.cli.trainer.check_param_init',
    ) as check:
        with pytest.raises(ValueError, match='source construction failed') as exc:
            trainer._setup()
        assert exc.value is error
        check.assert_not_called()


def test_check_collective_hashing_error():
    """Report a local hashing failure through the collective rather than exiting early.

    A meta buffer cannot be copied to CPU for hashing. Mock four gathered
    payloads with this failure at rank 1, then require one collective call and
    an error identifying rank 1.
    """
    model = _plain_model()
    reference = _payload(model)
    broken = torch.nn.Module()
    broken.register_buffer('unmaterialized', torch.empty(2, device='meta'))
    with patch('nnscaler.cli.trainer.dist.is_initialized', return_value=True), patch(
        'nnscaler.cli.trainer.dist.get_world_size', return_value=4,
    ), patch('nnscaler.cli.trainer.dist.all_gather_object') as gather:
        gather.side_effect = lambda outputs, value: outputs.__setitem__(
            slice(None), [reference, value, reference, reference],
        )
        with pytest.raises(RuntimeError, match='initialization failed: rank 1:'):
            check_param_init(broken)
        gather.assert_called_once()
