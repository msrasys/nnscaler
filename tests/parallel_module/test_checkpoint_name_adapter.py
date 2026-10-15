#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy
from dataclasses import replace

import pytest
import torch
from torch import nn

import nnscaler
from nnscaler.parallel import ComputeConfig
from nnscaler.runtime.module import AttrMeta, OriginModuleMetadata, ParallelModule, ZeroMetadata

from .common import assert_equal


class _ParallelModuleStub(ParallelModule, skip_init=True):
    # Real compilation and loading are covered by the CLI resume tests.
    def __init__(self, tid, *, width=2, dtype=torch.float32):
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
        self._register_state_dict_hook(self._save_metadata)

    @staticmethod
    def _save_metadata(module, state_dict, prefix, local_metadata):
        module._add_extra_state(state_dict, prefix)


def test_rename_parameters_and_buffers_without_copying_tensors():
    old = _ParallelModuleStub(10)
    new = _ParallelModuleStub(100)
    state = old.state_dict()
    before = copy.deepcopy(state)

    adapted = nnscaler.align_state_dict_names(new, state)

    assert set(adapted) == {'weight_100', 'bias_101', 'buffer_102', ParallelModule.EXTRA_STATE_KEY}
    assert adapted['weight_100'] is state['weight_10']
    assert adapted['bias_101'] is state['bias_11']
    assert adapted['buffer_102'] is state['buffer_12']
    assert_equal(adapted[ParallelModule.EXTRA_STATE_KEY], new.state_dict()[ParallelModule.EXTRA_STATE_KEY])
    assert adapted._metadata is state._metadata
    assert_equal(state, before)


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


def test_reject_changed_parameter_shape():
    old = _ParallelModuleStub(10, width=2)
    new = _ParallelModuleStub(100, width=4)

    with pytest.raises(ValueError, match='expected one match'):
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


def test_reject_changed_parameter_dtype():
    old = _ParallelModuleStub(10, dtype=torch.float32)
    new = _ParallelModuleStub(100, dtype=torch.float64)

    with pytest.raises(ValueError, match='expected one match'):
        nnscaler.align_state_dict_names(new, old.state_dict())


@pytest.mark.parametrize('source', ['checkpoint', 'module'])
def test_reject_two_shards_with_same_original_name(source):
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

    with pytest.raises(AssertionError, match=f'Duplicate original names in {source} param_area_map'):
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
