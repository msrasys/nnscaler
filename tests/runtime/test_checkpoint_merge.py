#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import gc
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import asdict

import pytest
import torch

import nnscaler
from nnscaler.cli.trainer import Trainer
from nnscaler.parallel import ComputeConfig, ModuleParameterLocation, OptimizerExtraState
from nnscaler.runtime.checkpoint import CheckpointMerger, _DeferredMerge, _StorageWriter
from nnscaler.runtime.module import AttrMeta, ExtraState, Zero3AttrMeta


pytestmark = pytest.mark.skipif(
    tuple(int(v) for v in torch.__version__.split('.')[:2]) < (2, 1),
    reason='tensor checkpoint merging requires PyTorch >= 2.1',
)


def assert_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_equal(left, right)
    else:
        assert actual == expected


def make_checkpoints(directory, *, zero=1, dim=0, dtype=torch.float32, optimizer='AdamW',
                     rows=8, columns=8, dedup=False, empty_optimizer=False):
    directory.mkdir()
    weight = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns).to(dtype)
    config = ComputeConfig(2, 4, use_zero=zero)
    files = []
    for rank in range(4):
        tp, dp = rank % 2, rank // 2
        slices = [slice(0, rows), slice(0, columns)]
        width = weight.shape[dim] // 2
        slices[dim] = slice(tp * width, (tp + 1) * width)
        slices = tuple(slices)
        shard = weight[slices].contiguous()
        numel = shard.numel()
        chunk = numel // 2
        state_names = ('exp_avg', 'exp_avg_sq', 'fp32_params') if optimizer == 'AdamW' else ('momentum_buffer',)
        states = {name: shard.float() + index for index, name in enumerate(state_names)}
        zero3 = None
        if zero:
            states = {name: value.flatten()[dp * chunk:(dp + 1) * chunk].clone()
                      for name, value in states.items()}
        if zero == 3:
            local = shard.flatten()[dp * chunk:(dp + 1) * chunk].clone()
            zero3 = {'weight_0': Zero3AttrMeta('weight', 'weight_0', dp * chunk, (dp + 1) * chunk, chunk)}
        else:
            local = shard
        if zero == 1:
            index_map = {0: (0, 0, numel, shard.shape)}
        elif zero == 3:
            index_map = {0: (0, 0, chunk, local.shape)}
        else:
            index_map = None
        fullmap = {
            # Buffers precede parameters to exercise optimizer local ID filtering.
            'buffer_1': AttrMeta(1, False, 'buffer', (2,), (slice(0, 2),), 1, torch.int64, (2,)),
            'weight_0': AttrMeta(0, True, 'weight', weight.shape, slices, 1, dtype, shard.shape),
        }
        extra = ExtraState(
            rank=rank, compute_config=config, dist_param_map={}, param_area_map=fullmap,
            cube_param_names=['weight_0'],
            origin_state_dict_names=['buffer', 'weight', 'tied_weight', 'unused'],
            origin_param_names=['weight', 'unused'],
            origin_shared_param_names=[['weight', 'tied_weight']],
            model_idx2opt_idx=index_map,
            opt_idx2ranks={0: ([tp, tp + 2], numel)} if zero else None,
            zero=zero, zero3_param_metadata=zero3,
        )
        opt_extra = OptimizerExtraState(
            rank, optimizer, {'backbone': ModuleParameterLocation(0, 1)}, {'backbone': config},
        )
        model = {
            'backbone.CUBE_EXTRA_STATE': asdict(extra),
            'backbone.weight_0': local,
            'backbone.buffer_1': torch.tensor([3, 5]),
            'outside': torch.tensor([9.0]),
        }
        if dedup and rank >= config.module_dedup_group_size:
            model.pop('backbone.weight_0')
            model.pop('backbone.buffer_1')
        states['step'] = torch.tensor(10.0)
        if not zero:
            states['tag'] = 7
        opt = {
            'CUBE_EXTRA_STATE': asdict(opt_extra),
            'state': {} if empty_optimizer or (dedup and rank >= config.optimizer_dedup_group_size) else {0: states},
            'param_groups': [{'params': [0], 'lr': 0.01, 'betas': (0.9, 0.999)}],
        }
        checkpoint = {
            'model': model, 'optimizer': opt, 'rank': rank,
            'train_args': {'model': {'type': 'fixture'}, 'checkpoint': {'save_type': 'deduped' if dedup else 'sharded'}},
            'train_status': {'finished_train_steps': 10},
            'lr_scheduler': {'last_epoch': 10}, 'nnscaler': 'test',
            'rng_states': torch.arange(16, dtype=torch.uint8),
            'dataloader': {'rank': rank, 'step': 10},
        }
        path = directory / f'{rank}.ckpt'
        torch.save(checkpoint, path)
        files.append(path)
    return files, weight


@pytest.mark.parametrize('zero', [0, 1, 3])
@pytest.mark.parametrize('dim', [0, 1])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('optimizer', ['AdamW', 'Muon'])
@pytest.mark.parametrize('dedup', [False, True])
def test_trainer_tensor_merge(tmp_path, zero, dim, dtype, optimizer, dedup):
    files, weight = make_checkpoints(tmp_path / 'input', zero=zero, dim=dim, dtype=dtype,
                                    optimizer=optimizer, dedup=dedup)
    hashes = [hashlib.sha256(path.read_bytes()).digest() for path in files]
    reference = Trainer._merge_checkpoint(files)
    path = tmp_path / 'merged.ckpt'
    Trainer.merge_checkpoint(files[::-1], path, merge_policy='tensor', ram_budget_bytes=8192)
    actual = torch.load(path, weights_only=False, mmap=True)
    assert_equal(actual, reference)
    assert torch.equal(actual['model']['backbone.weight'], weight)
    assert actual['model']['backbone.weight'].data_ptr() == actual['model']['backbone.tied_weight'].data_ptr()
    assert actual['optimizer']['param_groups'][0]['params'] == [0, 1]
    assert actual['train_args']['checkpoint']['save_type'] == 'merged'
    assert [state['rank'] for state in actual['dataloader']] == list(range(4))
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    assert hashes == [hashlib.sha256(path.read_bytes()).digest() for path in files]
    assert not list(tmp_path.glob('.*.tmp'))


@pytest.mark.parametrize('zero', [0, 1, 3])
@pytest.mark.parametrize('model_only,empty_optimizer', [(True, False), (False, True)])
def test_model_only_and_uninitialized_optimizer(tmp_path, zero, model_only, empty_optimizer):
    files, _ = make_checkpoints(tmp_path / 'input', zero=zero, empty_optimizer=empty_optimizer)
    expected = Trainer._merge_checkpoint(files, model_only=model_only)
    Trainer.merge_checkpoint(files, tmp_path / 'out.pt', model_only=model_only, merge_policy='tensor')
    assert_equal(torch.load(tmp_path / 'out.pt', weights_only=False), expected)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.int64, torch.bool])
def test_versions_strides_scalars_empty_aliases(tmp_path, dtype):
    source = torch.arange(16).reshape(4, 4).to(dtype)

    def merge():
        target = torch.zeros(source.shape, dtype=dtype)
        target[:2].copy_(source[:2])
        snapshot = target.clone()
        target[2:].copy_(source[2:])
        holes = torch.empty_strided((4, 4), (8, 1), dtype=dtype)
        holes.copy_(source)
        return {
            'target': target, 'snapshot': snapshot, 'alias': target.t(),
            'offset': target[1:, 1:], 'holes': holes,
            'scalar': target[0, 0].clone(), 'empty': torch.empty((0, 3), dtype=dtype),
        }

    expected = merge()
    merger = CheckpointMerger(ram_budget_bytes=512)
    stats = merger.merge(merge, tmp_path / 'out.pt')
    assert stats.live_bytes == 0
    assert stats.peak_bytes <= 512
    actual = torch.load(tmp_path / 'out.pt', weights_only=False)
    assert_equal(actual, expected)
    assert actual['target'].untyped_storage()._cdata == actual['offset'].untyped_storage()._cdata
    assert actual['target'].untyped_storage()._cdata == actual['alias'].untyped_storage()._cdata


@pytest.mark.parametrize('dim', [0, 1])
def test_storage_budget_and_release(tmp_path, dim):
    files, _ = make_checkpoints(tmp_path / 'input', rows=512, columns=512, dim=dim)
    size = 512 * 512 * 4
    merger = CheckpointMerger(ram_budget_bytes=2 * size)
    states = [merger.load(path) for path in files]

    def merge():
        return nnscaler.merge_state_dicts(
            [state['model'] for state in states], [state['optimizer'] for state in states],
        )

    stats = merger.merge(merge, tmp_path / 'out.pt')
    assert stats.live_bytes == 0
    # One full optimizer tensor plus one TP-local ZeRO reconstruction.
    assert stats.peak_bytes == size + size // 2
    assert stats.storages >= 4
    merger.ram_budget_bytes = size
    existing = (tmp_path / 'out.pt').read_bytes()
    with pytest.raises(MemoryError, match='ram_budget_bytes'):
        merger.merge(merge, tmp_path / 'out.pt')
    gc.collect()
    assert merger.stats.live_bytes == 0
    assert (tmp_path / 'out.pt').read_bytes() == existing
    assert not list(tmp_path.glob('.*.tmp'))


@pytest.mark.parametrize('failure', ['copy', 'write', 'fsync', 'publish'])
def test_failure_does_not_publish(tmp_path, monkeypatch, failure):
    from nnscaler.runtime import checkpoint

    merger = CheckpointMerger()
    output = tmp_path / 'output.pt'
    output.write_bytes(b'previous checkpoint')
    source = torch.ones(8)

    def merge():
        return source.clone()

    def fail(*args, **kwargs):
        raise OSError('injected failure')

    if failure == 'copy':
        monkeypatch.setattr(_DeferredMerge, '_materialize', fail)
    elif failure == 'write':
        original_init = _StorageWriter.__init__

        class FailingWriter:
            def __init__(self, writer):
                self.writer = writer

            def __getattr__(self, name):
                return getattr(self.writer, name)

            def write_record(self, name, data, size):
                if isinstance(data, torch.UntypedStorage):
                    assert merger.stats.live_bytes > 0
                    fail()
                return self.writer.write_record(name, data, size)

        def initialize(self, writer, merge):
            original_init(self, FailingWriter(writer), merge)

        monkeypatch.setattr(_StorageWriter, '__init__', initialize)
    elif failure == 'fsync':
        monkeypatch.setattr(checkpoint.os, 'fsync', fail)
    else:
        monkeypatch.setattr(checkpoint.os, 'replace', fail)
    with pytest.raises(OSError, match='injected failure'):
        merger.merge(merge, output)
    gc.collect()
    assert merger.stats.live_bytes == 0
    assert output.read_bytes() == b'previous checkpoint'
    assert not list(tmp_path.glob('.*.tmp'))


@pytest.mark.parametrize('bad_state', ['model', 'optimizer', 'coverage', 'model_config', 'scheduler'])
def test_original_validation_is_preserved(tmp_path, bad_state):
    files, _ = make_checkpoints(tmp_path / 'input', zero=3 if bad_state == 'coverage' else 0)
    state = torch.load(files[1], weights_only=False)
    if bad_state == 'model':
        # Replicated buffer conflict within the first scale unit.
        state['model']['backbone.buffer_1'][0] += 1
    elif bad_state == 'optimizer':
        state['optimizer']['state'] = {}
    elif bad_state == 'coverage':
        state['model']['backbone.CUBE_EXTRA_STATE']['zero3_param_metadata']['weight_0']['end'] -= 1
    elif bad_state == 'model_config':
        state['train_args']['model']['type'] = 'different'
    else:
        state['lr_scheduler']['last_epoch'] = 11
    torch.save(state, files[1])
    with pytest.raises(ValueError) as eager:
        Trainer._merge_checkpoint(files)
    with pytest.raises(ValueError) as streamed:
        Trainer.merge_checkpoint(files, tmp_path / 'out.pt', merge_policy='tensor')
    assert str(streamed.value) == str(eager.value)
    assert not (tmp_path / 'out.pt').exists()


@pytest.mark.parametrize('kwargs', [
    {'merge_policy': 'unknown'},
    {'ram_budget_bytes': 1},
    {'merge_policy': 'tensor', 'ram_budget_bytes': -1},
    {'merge_policy': 'tensor', 'serializer': ''},
    {'merge_policy': 'tensor', 'serializer_args': {}},
])
def test_invalid_options(tmp_path, kwargs):
    with pytest.raises(ValueError):
        Trainer.merge_checkpoint([], tmp_path / 'out.pt', **kwargs)
    assert not (tmp_path / 'out.pt').exists()


def test_empty_input_and_legacy_format(tmp_path):
    with pytest.raises(ValueError, match='empty'):
        Trainer.merge_checkpoint([], tmp_path / 'out.pt', merge_policy='tensor')
    old = tmp_path / 'legacy.pt'
    torch.save({'a': torch.ones(1)}, old, _use_new_zipfile_serialization=False)
    with pytest.raises(RuntimeError, match='mmap'):
        CheckpointMerger().load(old)
    with pytest.raises(ValueError, match='safetensors'):
        Trainer.merge_checkpoint([], tmp_path / 'out.safetensors', merge_policy='tensor')


def test_unsupported_operation_and_input_mutation(tmp_path):
    source = torch.ones(8)
    merger = CheckpointMerger()
    with pytest.raises(RuntimeError, match='Unsupported operation'):
        merger.merge(lambda: source + source, tmp_path / 'out.pt')
    with pytest.raises(RuntimeError, match='cannot modify an input'):
        merger.merge(lambda: source.copy_(source.clone()), tmp_path / 'out.pt')
    assert torch.equal(source, torch.ones(8))
    assert not (tmp_path / 'out.pt').exists()


def test_split_payloads_and_reuse(tmp_path):
    files, _ = make_checkpoints(tmp_path / 'input')
    merger = CheckpointMerger()
    # A custom layout may mmap separate model/optimizer payloads with the same API.
    states = [merger.load(path) for path in files]
    models = [state['model'] for state in states]
    opts = [state['optimizer'] for state in states]
    expected = nnscaler.merge_state_dicts(models, opts)
    merger.merge(lambda: nnscaler.merge_state_dicts(models)[0], tmp_path / 'model.pt')
    assert merger.stats.live_bytes == 0
    merger.merge(lambda: nnscaler.merge_state_dicts(models, opts)[1], tmp_path / 'optimizer.pt')
    assert merger.stats.live_bytes == 0
    assert_equal(torch.load(tmp_path / 'model.pt', weights_only=False), expected[0])
    assert_equal(torch.load(tmp_path / 'optimizer.pt', weights_only=False), expected[1])


@pytest.mark.skipif(sys.platform != 'linux', reason='input eviction uses Linux mmap ranges')
def test_input_mappings_are_not_retained(tmp_path):
    path = tmp_path / 'input.pt'
    torch.save({'data': torch.ones(8)}, path)
    merger = CheckpointMerger()
    state = merger.load(path)
    assert len(merger._inputs) == 1
    del state
    gc.collect()
    assert not merger._inputs


def _restore_heap_tensor():
    return torch.ones(8)


class _HeapTensor:
    def __reduce__(self):
        return _restore_heap_tensor, ()


def test_heap_tensor_from_pickle_is_never_evicted(tmp_path):
    path = tmp_path / 'input.pt'
    torch.save({'data': _HeapTensor()}, path)
    merger = CheckpointMerger()
    state = merger.load(path)
    assert not merger._inputs
    merger.merge(lambda: state['data'].clone(), tmp_path / 'out.pt')
    assert torch.equal(state['data'], torch.ones(8))
    assert_equal(torch.load(tmp_path / 'out.pt', weights_only=False), state['data'])


def test_sync_count_does_not_scale_with_storages(tmp_path, monkeypatch):
    from nnscaler.runtime import checkpoint

    calls = []
    original = checkpoint.os.fsync

    def fsync(fd):
        calls.append(fd)
        return original(fd)

    monkeypatch.setattr(checkpoint.os, 'fsync', fsync)
    merger = CheckpointMerger()
    merger.merge(lambda: [torch.zeros(8) for _ in range(100)], tmp_path / 'out.pt')
    assert len(calls) == 1
    assert merger.stats.peak_bytes == 8 * 4
    assert merger.stats.live_bytes == 0


def test_unsupported_pytorch_version(monkeypatch):
    monkeypatch.setattr(torch, '__version__', '2.0.1')
    with pytest.raises(RuntimeError, match='PyTorch >= 2.1'):
        CheckpointMerger()


def test_budget_failure_releases_dependency_with_live_traceback(tmp_path):
    source = torch.ones(16)
    merger = CheckpointMerger(ram_budget_bytes=source.numel() * source.element_size())

    def merge():
        return source.clone().clone()

    with pytest.raises(MemoryError) as error:
        merger.merge(merge, tmp_path / 'out.pt')
    assert error.value.__traceback__ is not None
    assert merger.stats.live_bytes == 0


def test_comparison_budget_failure_releases_tensors(tmp_path):
    source = torch.ones(8)
    merger = CheckpointMerger(ram_budget_bytes=32)

    def merge():
        return torch.equal(source.clone(), source.clone())

    with pytest.raises(MemoryError) as error:
        merger.merge(merge, tmp_path / 'out.pt')
    assert error.value.__traceback__ is not None
    assert merger.stats.live_bytes == 0


def test_resume_adam_step(tmp_path):
    files, _ = make_checkpoints(tmp_path / 'input', zero=3, dim=1)
    reference = Trainer._merge_checkpoint(files)
    Trainer.merge_checkpoint(files, tmp_path / 'out.pt', merge_policy='tensor')
    actual = torch.load(tmp_path / 'out.pt', weights_only=False)
    resumed = []
    for checkpoint in (reference, actual):
        weight = torch.nn.Parameter(checkpoint['model']['backbone.weight'].clone())
        unused = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([weight, unused])
        group = dict(optimizer.state_dict()['param_groups'][0])
        group.update(checkpoint['optimizer']['param_groups'][0])
        optimizer.load_state_dict({'state': checkpoint['optimizer']['state'], 'param_groups': [group]})
        weight.square().sum().backward()
        optimizer.step()
        resumed.append((weight.detach(), optimizer.state_dict()))
    assert_equal(*resumed)


@pytest.mark.skipif(sys.platform != 'linux', reason='RSS high water and mmap eviction test uses Linux')
def test_peak_rss_reduction(tmp_path):
    # VmHWM belongs to the exec'd address space; getrusage can inherit the parent's
    # high water, masking the difference after other memory-intensive tests.
    files, _ = make_checkpoints(tmp_path / 'input', rows=2048, columns=2048)
    program = """
import json, sys, torch
from pathlib import Path
from nnscaler.cli.trainer import Trainer
torch.set_num_threads(1)
Trainer.merge_checkpoint(sys.argv[3:], sys.argv[2], merge_policy=sys.argv[1])
peak = next(int(line.split()[1]) for line in Path('/proc/self/status').read_text().splitlines()
            if line.startswith('VmHWM:'))
print(json.dumps({'peak_kib': peak}))
"""
    results = {}
    for policy in ('all', 'tensor'):
        result = subprocess.run(
            [sys.executable, '-c', program, policy, str(tmp_path / f'{policy}.pt'), *map(str, files)],
            text=True, capture_output=True, check=True, timeout=120,
            env={**os.environ, 'OMP_NUM_THREADS': '1'},
        )
        results[policy] = json.loads(result.stdout.splitlines()[-1])['peak_kib']
    # The fixture has 64 MiB of merged model/optimizer tensors, plus eager inputs.
    assert results['all'] - results['tensor'] >= 32 * 1024, results
    assert_equal(
        torch.load(tmp_path / 'tensor.pt', weights_only=False),
        torch.load(tmp_path / 'all.pt', weights_only=False),
    )
    print(f'Peak RSS (KiB): {results}')
