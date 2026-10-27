"""Gradient scaling must not read a device scalar into Python control flow."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from nnscaler.runtime.f16_optimizer import MixedPrecisionF16OptimizerMixin
from nnscaler.runtime.hybrid_optimizer import ScaleDelayedOptimizerMixin


def no_scalar_read(*args, **kwargs):
    raise AssertionError('gradient scaling read a tensor scalar on the host')


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('factor', [1.0, 0.375, 0.0, float('nan')])
@pytest.mark.parametrize('tensor_factor', [False, True])
def test_master_grad_copy_and_scale_match_reference(dtype, factor, tensor_factor):
    params = [torch.nn.Parameter(torch.arange(8, dtype=dtype)) for _ in range(4)]
    params[0].grad = torch.linspace(-1, 1, 8, dtype=dtype)
    params[1].grad = torch.ones(8, dtype=dtype)
    params[3].requires_grad_(False)
    masters = [torch.nn.Parameter(p.detach().float().clone()) for p in params]
    masters[1].grad = torch.full((8,), -9.0)
    reused = masters[1].grad
    masters[3].grad = torch.full((8,), 7.0)
    multiplier = torch.tensor(factor) if tensor_factor else factor
    expected = [(p.grad.float() if p.grad is not None else torch.zeros(8)) * multiplier
                for p in params[:3]]
    state = SimpleNamespace(f16_params=params, fp32_params=masters, _multiply_factor=multiplier)
    with patch.object(torch.Tensor, '__bool__', no_scalar_read), \
         patch.object(torch.Tensor, 'item', no_scalar_read):
        MixedPrecisionF16OptimizerMixin._sync_f16_grads_to_fp32(state)
    for p, want in zip(masters[:3], expected):
        torch.testing.assert_close(p.grad, want, rtol=0, atol=0, equal_nan=True)
    assert masters[1].grad is reused
    assert torch.equal(masters[3].grad, torch.full((8,), 7.0))
    assert state._multiply_factor == 1.0


@pytest.mark.parametrize('factor', [1.0, 0.375, 0.0])
def test_delayed_scaling_reads_no_scalar_and_updates_once(factor):
    params = [torch.nn.Parameter(torch.arange(8, dtype=torch.float32)) for _ in range(2)]
    params[0].grad = torch.linspace(-1, 1, 8)
    optimizer = torch.optim.SGD(params, lr=0.125)
    state = ScaleDelayedOptimizerMixin.apply_mixin(optimizer)
    state._step = optimizer.step
    state._multiply_factor = torch.tensor(factor)
    expected = params[0].detach() - 0.125 * params[0].grad * factor
    with patch.object(torch.Tensor, '__bool__', no_scalar_read), \
         patch.object(torch.Tensor, 'item', no_scalar_read):
        state.override_step()
    torch.testing.assert_close(params[0], expected, rtol=0, atol=0)
    assert params[1].grad is None
    assert state._multiply_factor == 1.0
