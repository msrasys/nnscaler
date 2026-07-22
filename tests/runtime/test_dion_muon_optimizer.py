#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from copy import deepcopy

import pytest
import torch


try:
    from dion.muon import Muon as _DionMuon  # noqa: F401
except ImportError:
    pytest.skip("Dion Muon not available", allow_module_level=True)


from nnscaler.runtime.adapter.reducer import FlattenParamInfo, ReducerParamInfo
from nnscaler.runtime.dion_optimizer import Muon as DionMuon
from nnscaler.runtime.f16_optimizer import MixedPrecisionDionMuon
from nnscaler.runtime.hybrid_optimizer import HybridOptimizer
from nnscaler.runtime.utils import set_fparam_meta


def _identity_newton_schulz(tensor, epsilon=1e-7):
    del epsilon
    return tensor


def _make_flattened_param(
    tensors,
    *,
    dtype,
    device,
    padding=0,
    opt_num_chunks=1,
    opt_chunk_index=0,
):
    embedded_params = [
        torch.nn.Parameter(tensor.to(device=device, dtype=dtype))
        for tensor in tensors
    ]
    params_info = {}
    offset = 0
    for param in embedded_params:
        end = offset + param.numel()
        params_info[param] = ReducerParamInfo(
            shape=param.shape,
            start=0,
            end=param.numel(),
            bucket_param_buffer_start=offset,
            bucket_param_buffer_end=end,
        )
        offset = end

    opt_numel = offset + padding
    assert opt_numel % opt_num_chunks == 0
    flatten_info = FlattenParamInfo(
        zero=1,
        params_info=params_info,
        opt_numel=opt_numel,
        opt_num_chunks=opt_num_chunks,
        opt_chunk_index=opt_chunk_index,
    )
    flat_param = torch.nn.Parameter(
        torch.zeros(
            flatten_info.opt_chunk_size,
            dtype=dtype,
            device=device,
        )
    )
    set_fparam_meta(flat_param, flatten_info)
    return flat_param, embedded_params


def _make_mixed_optimizer(tensors, device):
    flat_param, model_params = _make_flattened_param(
        tensors,
        dtype=torch.bfloat16,
        device=device,
    )
    optimizer = MixedPrecisionDionMuon(
        [flat_param],
        lr=0.01,
        use_triton=False,
        use_polar_express=False,
        newton_schulz_func=_identity_newton_schulz,
    )
    return model_params, optimizer


def _populate_fp32_state(optimizer):
    for index, fp32_param in enumerate(optimizer.fp32_params):
        fp32_param.data.add_(0.125 * (index + 1))
        optimizer.state[fp32_param]['momentum'] = torch.full_like(
            fp32_param,
            0.25 * (index + 1),
        )


@pytest.mark.parametrize(
    'device',
    [
        'cpu',
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason='CUDA required',
            ),
        ),
    ],
)
def test_mixed_precision_checkpoint_dump_and_load_are_fp32(tmp_path, device):
    torch.manual_seed(0)
    tensors = [torch.randn(4, 8), torch.randn(8, 4)]
    model_params, optimizer = _make_mixed_optimizer(tensors, device)
    _populate_fp32_state(optimizer)

    checkpoint_path = tmp_path / 'dion_muon.pt'
    torch.save(
        {
            'model': [param.detach().cpu() for param in model_params],
            'optimizer': optimizer.state_dict(),
        },
        checkpoint_path,
    )
    assert all(
        'fp32_params' not in optimizer.state[param]
        for param in optimizer.fp32_params
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False,
    )

    assert all(param.dtype == torch.bfloat16 for param in checkpoint['model'])
    state = checkpoint['optimizer']['state'][0]
    assert set(state) == {'momentum', 'fp32_params'}
    assert state['momentum'].dtype == torch.float32
    assert state['fp32_params'].dtype == torch.float32

    restored_model_params, restored = _make_mixed_optimizer(
        checkpoint['model'],
        device,
    )
    restored.load_state_dict(deepcopy(checkpoint['optimizer']))

    assert all(param.dtype == torch.bfloat16 for param in restored_model_params)
    assert all(param.dtype == torch.float32 for param in restored.fp32_params)
    assert all(
        restored.state[param]['momentum'].dtype == torch.float32
        for param in restored.fp32_params
    )
    for expected, actual in zip(optimizer.fp32_params, restored.fp32_params):
        torch.testing.assert_close(actual.cpu(), expected.cpu())


def test_legacy_bf16_state_is_normalized_to_fp32():
    tensors = [torch.randn(4, 8), torch.randn(8, 4)]
    _, optimizer = _make_mixed_optimizer(tensors, 'cpu')
    _populate_fp32_state(optimizer)
    legacy_state = deepcopy(optimizer.state_dict())
    entry = legacy_state['state'][0]
    entry['momentum_buffer'] = entry.pop('momentum').bfloat16()
    entry['fp32_params'] = entry['fp32_params'].bfloat16()

    _, restored = _make_mixed_optimizer(tensors, 'cpu')
    restored.load_state_dict(legacy_state)

    # Loading must not mutate a checkpoint object that may be reused elsewhere.
    assert 'momentum_buffer' in legacy_state['state'][0]
    assert legacy_state['state'][0]['fp32_params'].dtype == torch.bfloat16
    for fp32_param in restored.fp32_params:
        assert fp32_param.dtype == torch.float32
        assert restored.state[fp32_param]['momentum'].dtype == torch.float32

    normalized = restored.state_dict()['state'][0]
    assert 'momentum_buffer' not in normalized
    assert normalized['momentum'].dtype == torch.float32
    assert normalized['fp32_params'].dtype == torch.float32


def test_legacy_momentum_key_loads_into_dion_muon():
    param = torch.nn.Parameter(torch.randn(4, 8))
    optimizer = DionMuon(
        [param],
        use_triton=False,
        use_polar_express=False,
    )
    optimizer.state[param]['momentum_buffer'] = torch.randn_like(param)
    state_dict = deepcopy(optimizer.state_dict())
    assert set(state_dict['state'][0]) == {'momentum'}
    expected = state_dict['state'][0].pop('momentum')
    state_dict['state'][0]['momentum_buffer'] = expected

    restored_param = torch.nn.Parameter(torch.randn(4, 8))
    restored = DionMuon(
        [restored_param],
        use_triton=False,
        use_polar_express=False,
    )
    restored.load_state_dict(state_dict)

    torch.testing.assert_close(
        restored.state[restored_param]['momentum'],
        expected,
    )
    assert 'momentum_buffer' in state_dict['state'][0]


def test_padding_only_zero1_chunk_has_stable_checkpoint_layout():
    flat_param, _ = _make_flattened_param(
        [torch.zeros(2, 4)],
        dtype=torch.bfloat16,
        device='cpu',
        padding=8,
        opt_num_chunks=2,
        opt_chunk_index=1,
    )

    for optimizer_type in (DionMuon, MixedPrecisionDionMuon):
        optimizer = optimizer_type(
            [flat_param],
            use_triton=False,
            use_polar_express=False,
        )
        state_dict = optimizer.state_dict()
        expected_keys = {'momentum'} if optimizer_type is DionMuon else {'momentum', 'fp32_params'}
        assert set(state_dict['state']) == {0} and set(state_dict['state'][0]) == expected_keys
        assert state_dict['param_groups'][0]['params'] == [0] and optimizer.param_groups[0]['params'] == []
        optimizer.load_state_dict(deepcopy(state_dict))


def test_hybrid_optimizer_checkpoint_round_trip():
    tensors = [torch.randn(4, 8), torch.randn(8, 4)]
    flat_param, _ = _make_flattened_param(
        tensors,
        dtype=torch.bfloat16,
        device='cpu',
    )
    config = {
        'optimizers': [{
            'type': MixedPrecisionDionMuon,
            'options': {
                'lr': 0.01,
                'use_triton': False,
                'use_polar_express': False,
                'newton_schulz_func': _identity_newton_schulz,
            },
        }],
    }
    optimizer = HybridOptimizer(
        [flat_param],
        {flat_param: (0, 0)},
        config,
    )
    _populate_fp32_state(optimizer.optimizers[0])
    state_dict = deepcopy(optimizer.state_dict())

    assert state_dict['state'][0]['momentum'].dtype == torch.float32
    assert state_dict['state'][0]['fp32_params'].dtype == torch.float32

    restored_flat_param, _ = _make_flattened_param(
        tensors,
        dtype=torch.bfloat16,
        device='cpu',
    )
    restored = HybridOptimizer(
        [restored_flat_param],
        {restored_flat_param: (0, 0)},
        config,
    )
    restored.load_state_dict(state_dict)

    restored_dion = restored.optimizers[0]
    assert all(param.dtype == torch.float32 for param in restored_dion.fp32_params)
    assert all(
        restored_dion.state[param]['momentum'].dtype == torch.float32
        for param in restored_dion.fp32_params
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_resume_next_step_matches_uninterrupted_training():
    torch.manual_seed(42)
    tensors = [torch.randn(8, 8), torch.randn(8, 8)]
    model_params, optimizer = _make_mixed_optimizer(tensors, 'cuda')

    for param in model_params:
        param.grad = torch.randn_like(param)
    optimizer.step()
    checkpoint = {
        'model': [param.detach().cpu() for param in model_params],
        'optimizer': deepcopy(optimizer.state_dict()),
    }

    restored_model_params, restored = _make_mixed_optimizer(
        checkpoint['model'],
        'cuda',
    )
    restored.load_state_dict(deepcopy(checkpoint['optimizer']))

    torch.manual_seed(123)
    grads = [torch.randn_like(param) for param in model_params]
    for param, restored_param, grad in zip(
        model_params,
        restored_model_params,
        grads,
    ):
        param.grad = grad.clone()
        restored_param.grad = grad.clone()

    optimizer.step()
    restored.step()

    for expected, actual in zip(model_params, restored_model_params):
        torch.testing.assert_close(actual, expected)
    for expected, actual in zip(optimizer.fp32_params, restored.fp32_params):
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            restored.state[actual]['momentum'],
            optimizer.state[expected]['momentum'],
        )
