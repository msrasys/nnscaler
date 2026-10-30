#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy
import io
from dataclasses import asdict, replace

import pytest
import torch
from torch import nn

import nnscaler
from nnscaler.parallel import ComputeConfig
from nnscaler.runtime.module import AttrMeta, ExtraState, OriginModuleMetadata, ParallelModule, Zero3AttrMeta, ZeroMetadata

from .common import assert_equal


class _ParallelModuleStub(ParallelModule, skip_init=True):
    # Real compilation and loading are covered by the CLI resume tests.
    def __init__(self, tid, *, width=2, dtype=torch.float32, buffer_shards=False):
        nn.Module.__init__(self)
        self.rank = 0
        self.compute_config = ComputeConfig(1, 1)
        self.dist_param_map = {'weight': 'weight', 'bias': 'bias', 'buffer': 'buffer'}
        self.origin_module_metadata = OriginModuleMetadata(
            origin_state_dict_names=['weight', 'bias', 'buffer'],
            origin_param_names=['weight', 'bias'],
            origin_shared_param_names=[],
        )
        self._zero_metadata = ZeroMetadata()
        self._fullmap = {}
        for index, name in enumerate(['weight', 'bias', 'buffer']):
            local_name = f'{name}_{tid + index}'
            tensor = torch.full((width,), float(index + 1), dtype=dtype)
            if name == 'buffer':
                self.register_buffer(local_name, tensor)
            else:
                self.register_parameter(local_name, nn.Parameter(tensor))
            self._fullmap[local_name] = AttrMeta(
                tid=tid + index, is_param=name != 'buffer',
                orig_name=name, shape=(width,), slicers=(slice(0, width),),
                val_chunks=1, dtype=dtype, sub_shape=(width,),
            )
        if buffer_shards:
            buffer_name = f'buffer_{tid + 2}'
            self.register_buffer(buffer_name + '_shard', getattr(self, buffer_name)[:1].clone())
            self._fullmap[buffer_name + '_shard'] = replace(
                self._fullmap[buffer_name], slicers=(slice(0, 1),), sub_shape=(1,),
            )
        self._register_state_dict_hook(self._save_metadata)

    @staticmethod
    def _save_metadata(module, state_dict, prefix, local_metadata):
        module._add_extra_state(state_dict, prefix)


@pytest.mark.parametrize('with_buffer_shard', [False, True])
def test_rename_parameters_and_buffers_without_copying_tensors(with_buffer_shard):
    old = _ParallelModuleStub(10, buffer_shards=with_buffer_shard)
    new = _ParallelModuleStub(100, buffer_shards=with_buffer_shard)
    state = old.state_dict()
    before = copy.deepcopy(state)

    adapted = nnscaler.align_state_dict_names(new, state)

    expected_names = {'weight_100', 'bias_101', 'buffer_102', ParallelModule.EXTRA_STATE_KEY}
    if with_buffer_shard:
        expected_names.add('buffer_102_shard')
        assert adapted['buffer_102_shard'] is state['buffer_12_shard']
    assert set(adapted) == expected_names
    assert adapted['weight_100'] is state['weight_10']
    assert adapted['bias_101'] is state['bias_11']
    assert adapted['buffer_102'] is state['buffer_12']
    assert_equal(adapted[ParallelModule.EXTRA_STATE_KEY], new.state_dict()[ParallelModule.EXTRA_STATE_KEY])
    assert adapted._metadata is state._metadata
    assert_equal(state, before)


@pytest.mark.parametrize('legacy', [False, True])
def test_zero3_metadata_round_trip(legacy):
    meta = Zero3AttrMeta('weight', 'weight_10', 0, 2, 2)
    mapping = {'weight_10': asdict(meta) if legacy else meta}
    if legacy:
        mapping['bias_11'] = None
    module = _ParallelModuleStub(10)
    module._zero_metadata = ZeroMetadata(zero=3, zero3_param_metadata=mapping)
    assert module._zero_metadata.zero3_param_metadata == {'weight_10': meta}

    state = module.state_dict()[ParallelModule.EXTRA_STATE_KEY]
    assert state['zero3_param_metadata'] == {'weight_10': asdict(meta)}
    state['zero3_param_metadata'] = mapping
    extra = ExtraState(**state)
    assert extra.zero3_param_metadata == {'weight_10': meta}
    assert isinstance(extra.param_area_map['weight_10'], AttrMeta)

    # Loading directly pickled legacy metadata must normalize it as well.
    extra.zero3_param_metadata = mapping
    checkpoint = io.BytesIO()
    torch.save(extra, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=False)
    assert restored.zero3_param_metadata == {'weight_10': meta}


@pytest.mark.parametrize('legacy', [False, True])
def test_merge_zero3_metadata(legacy):
    module = _ParallelModuleStub(10)
    fullmap = {'weight_10': module._fullmap['weight_10']}
    zero_idx_maps = []
    for rank in range(2):
        meta = Zero3AttrMeta('weight', 'weight_10', rank, rank + 1, 1)
        mapping = {'weight_10': asdict(meta) if legacy else meta}
        if legacy:
            mapping['bias_11'] = None
        # Entries contain model-to-optimizer indices, bucket information, ZeRO level, and shard metadata.
        zero_idx_maps.append((None, None, 3, mapping))
    merged = ParallelModule.merge_model_state_dicts(
        [{'weight_10': torch.tensor([1.0])}, {'weight_10': torch.tensor([2.0])}],
        [fullmap, fullmap], zero_idx_maps,
    )
    assert_equal(merged, {'weight': torch.tensor([1.0, 2.0])})


def test_rename_parallel_submodules_without_changing_regular_module():
    old = nn.ModuleDict({
        'linear': nn.Linear(2, 2),
        'encoder': _ParallelModuleStub(10),
        'decoder': _ParallelModuleStub(20),
    })
    new = nn.ModuleDict({
        'linear': nn.Linear(2, 2),
        'encoder': _ParallelModuleStub(100),
        'decoder': _ParallelModuleStub(200),
    })
    state = old.state_dict()

    adapted = nnscaler.align_state_dict_names(new, state)

    assert set(adapted) == set(new.state_dict())
    assert adapted['linear.weight'] is state['linear.weight']
    assert adapted['linear.bias'] is state['linear.bias']
    assert adapted['encoder.weight_100'] is state['encoder.weight_10']
    assert adapted['decoder.weight_200'] is state['decoder.weight_20']
    assert adapted['encoder.buffer_102'] is state['encoder.buffer_12']
    assert adapted['decoder.buffer_202'] is state['decoder.buffer_22']


@pytest.mark.parametrize('change', ['add', 'remove', 'shape', 'dtype'])
def test_reject_parameter_changes(change):
    old = _ParallelModuleStub(10)
    new = _ParallelModuleStub(
        100, width=4 if change == 'shape' else 2,
        dtype=torch.float64 if change == 'dtype' else torch.float32,
    )

    if change == 'add':
        new.register_parameter('extra_999', nn.Parameter(torch.ones(2)))
        new._fullmap['extra_999'] = replace(
            new._fullmap['weight_100'], tid=999, orig_name='extra',
        )
        new.dist_param_map['extra'] = 'extra'
        new.origin_module_metadata.origin_param_names.append('extra')
        new.origin_module_metadata.origin_state_dict_names.append('extra')
    elif change == 'remove':
        del new.bias_101
        del new._fullmap['bias_101']
        del new.dist_param_map['bias']
        new.origin_module_metadata.origin_param_names.remove('bias')
        new.origin_module_metadata.origin_state_dict_names.remove('bias')

    message = 'origin_param_names changed' if change in ('add', 'remove') else 'expected one match'
    with pytest.raises(ValueError, match=message):
        nnscaler.align_state_dict_names(new, old.state_dict())


def test_reject_different_shards_of_same_parameter():
    old = _ParallelModuleStub(10)
    new = _ParallelModuleStub(100)
    old._fullmap['weight_10'] = replace(
        old._fullmap['weight_10'], shape=(4,), slicers=(slice(0, 2),),
    )
    new._fullmap['weight_100'] = replace(
        new._fullmap['weight_100'], shape=(4,), slicers=(slice(2, 4),),
    )

    with pytest.raises(ValueError, match='expected one match'):
        nnscaler.align_state_dict_names(new, old.state_dict())


@pytest.mark.parametrize('kind', ['parameter', 'buffer'])
@pytest.mark.parametrize('slicers', [(), Ellipsis, '...'])
def test_align_scalar_tensor(kind, slicers):
    old, new = _ParallelModuleStub(10), _ParallelModuleStub(100)
    # Generated names for the old and newly traced scalar tensor.
    names = ('weight_10', 'weight_100') if kind == 'parameter' else ('buffer_12', 'buffer_102')
    for module, name in zip((old, new), names):
        delattr(module, name)
        tensor = torch.tensor(3.0)
        if kind == 'parameter':
            module.register_parameter(name, nn.Parameter(tensor))
        else:
            module.register_buffer(name, tensor)
        module._fullmap[name] = replace(
            module._fullmap[name], shape=(), sub_shape=(),
            slicers=slicers if module is old else (),
        )
        assert module._fullmap[name].slicers == ()

    state = old.state_dict()
    state[ParallelModule.EXTRA_STATE_KEY]['param_area_map'][names[0]]['slicers'] = slicers
    adapted = nnscaler.align_state_dict_names(new, state)
    assert adapted[names[1]] is state[names[0]]
    assert adapted[names[1]].shape == torch.Size([])
    assert state[ParallelModule.EXTRA_STATE_KEY]['param_area_map'][names[0]]['slicers'] == slicers

    # Newly saved metadata also loads into modules generated with the old slicers.
    state = new.state_dict()
    adapted = nnscaler.align_state_dict_names(old, state)
    assert adapted[names[0]] is state[names[1]]


@pytest.mark.parametrize('slicers', [Ellipsis, '...'])
def test_load_pickled_legacy_scalar_metadata(slicers):
    meta = AttrMeta(1, True, 'weight', (), (), 1, torch.float32, ())
    # Simulate an AttrMeta instance saved before scalar slicers were normalized.
    meta.slicers = slicers
    checkpoint = io.BytesIO()
    torch.save({'param_area_map': {'weight_1': meta}}, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, weights_only=False)['param_area_map']['weight_1']
    assert restored == replace(meta, slicers=())


@pytest.mark.parametrize('source', ['checkpoint', 'module'])
def test_reject_incompatible_parameter_shards(source):
    old = _ParallelModuleStub(10)
    new = _ParallelModuleStub(100)
    module = old if source == 'checkpoint' else new
    weight_name = 'weight_10' if source == 'checkpoint' else 'weight_100'
    module._fullmap[weight_name] = replace(
        module._fullmap[weight_name], shape=(4,), slicers=(slice(0, 2),),
    )
    module._fullmap['another_weight'] = replace(
        module._fullmap[weight_name], tid=999, slicers=(slice(2, 4),),
    )

    with pytest.raises(ValueError, match='expected one match'):
        nnscaler.align_state_dict_names(new, old.state_dict())


@pytest.mark.parametrize('source', ['checkpoint', 'module'])
def test_reject_ambiguous_buffer_shards(source):
    old = _ParallelModuleStub(10, buffer_shards=True)
    new = _ParallelModuleStub(100, buffer_shards=True)
    module = old if source == 'checkpoint' else new
    buffer_name = 'buffer_12' if source == 'checkpoint' else 'buffer_102'
    module._fullmap[buffer_name + '_shard'] = replace(module._fullmap[buffer_name], tid=999)

    message = 'expected one match' if source == 'checkpoint' else 'ambiguous shard metadata'
    with pytest.raises(ValueError, match=message):
        nnscaler.align_state_dict_names(new, old.state_dict())


def test_reject_checkpoint_without_parallel_metadata():
    old = _ParallelModuleStub(10)
    state = old.state_dict()
    del state[ParallelModule.EXTRA_STATE_KEY]

    with pytest.raises(ValueError, match='missing CUBE_EXTRA_STATE'):
        nnscaler.align_state_dict_names(_ParallelModuleStub(100), state)


def test_reject_changed_optimizer_parameter_order():
    old = _ParallelModuleStub(10)
    state = old.state_dict()
    state[ParallelModule.EXTRA_STATE_KEY]['cube_param_names'].reverse()

    with pytest.raises(ValueError, match='optimizer parameter order changed'):
        nnscaler.align_state_dict_names(_ParallelModuleStub(100), state)


def test_reject_changed_world_size():
    old = _ParallelModuleStub(10)
    new = _ParallelModuleStub(100)
    old.compute_config = ComputeConfig(1, 2)
    new.compute_config = ComputeConfig(1, 4)

    with pytest.raises(ValueError, match='runtime_ngpus changed'):
        nnscaler.align_state_dict_names(new, old.state_dict())


def test_nonparallel_checkpoint_unchanged():
    module = nn.Linear(2, 2)
    state = module.state_dict()
    adapted = nnscaler.align_state_dict_names(module, state)
    assert_equal(adapted, state)
    assert adapted['weight'] is state['weight']


def test_sharded_load_preserves_metadata():
    state = nn.BatchNorm1d(2).state_dict()
    del state['num_batches_tracked']
    with pytest.raises(RuntimeError, match='Missing key.*num_batches_tracked'):
        nnscaler.load_sharded_state_dict(nn.BatchNorm1d(2), state, device='cpu')
