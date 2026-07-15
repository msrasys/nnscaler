#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import math

import pytest
import torch

import nnscaler.runtime.dion_muon_optimizer as dion_muon_optimizer
from nnscaler.runtime.dion_muon_optimizer import cape_polar


def _zero_newton_schulz(tensor, epsilon=1e-7):
    del epsilon
    return torch.zeros_like(tensor)


@pytest.mark.parametrize(
    ('shape', 'dtype'),
    [
        ((8, 16), torch.bfloat16),
        ((16, 8), torch.float32),
        ((4, 8, 16), torch.bfloat16),
        ((2, 4, 8, 16), torch.float32),
    ],
)
def test_cape_shape_dtype_and_update_scale(monkeypatch, shape, dtype):
    monkeypatch.setattr(
        dion_muon_optimizer,
        'newton_schulz_triton',
        _zero_newton_schulz,
    )
    torch.manual_seed(0)
    gradient = torch.randn(shape, dtype=dtype)

    update = cape_polar(gradient)

    assert update.shape == gradient.shape
    assert update.dtype == dtype
    assert torch.isfinite(update).all()
    expected_norm = 1.2 * math.sqrt(min(shape[-2], shape[-1]))
    actual_norm = update.float().norm(dim=(-2, -1))
    torch.testing.assert_close(
        actual_norm,
        torch.full_like(actual_norm, expected_norm),
        rtol=5e-3,
        atol=5e-3,
    )


def test_cape_zero_input_stays_finite_and_zero(monkeypatch):
    monkeypatch.setattr(
        dion_muon_optimizer,
        'newton_schulz_triton',
        _zero_newton_schulz,
    )
    update = cape_polar(torch.zeros((4, 8, 16), dtype=torch.bfloat16))

    assert torch.isfinite(update).all()
    assert torch.count_nonzero(update) == 0


def test_cape_rejects_vector_input():
    with pytest.raises(AssertionError):
        cape_polar(torch.zeros(8))


def _reference_mclip(tensor, polar, radius, epsilon=1e-7):
    transposed = tensor.size(-2) < tensor.size(-1)
    if transposed:
        tensor, polar = tensor.mT, polar.mT
    scaled_polar = radius * polar
    gram = tensor.mT @ tensor
    identity = torch.eye(
        gram.size(-1),
        device=gram.device,
        dtype=gram.dtype,
    ).expand_as(gram)
    sign = dion_muon_optimizer.newton_schulz_triton(
        gram - radius * radius * identity,
        epsilon=epsilon,
    )
    clipped = 0.5 * (
        tensor + scaled_polar + (scaled_polar - tensor) @ sign
    )
    return clipped.mT if transposed else clipped


def _reference_cape(gradient, epsilon=1e-7, c=3.0, iterations=5):
    dtype = gradient.dtype
    iterate = gradient.float()
    rank = min(gradient.shape[-2], gradient.shape[-1])
    sqrt_rank = rank ** 0.5
    iterate = iterate / (
        iterate.norm(dim=(-2, -1), keepdim=True) * 1.02 + epsilon
    )
    radius = c / sqrt_rank
    polar = dion_muon_optimizer.newton_schulz_triton(
        iterate,
        epsilon=epsilon,
    )
    for _ in range(iterations):
        iterate = _reference_mclip(
            iterate,
            polar,
            radius,
            epsilon,
        )
        iterate = iterate / (
            iterate.norm(dim=(-2, -1), keepdim=True) * 1.02 + epsilon
        )
    iterate = iterate / (
        iterate.norm(dim=(-2, -1), keepdim=True) + epsilon
    )
    return (1.2 * sqrt_rank * iterate).to(dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('shape', [(8, 16), (2, 16, 8)])
def test_cape_matches_llm_train_reference(shape):
    pytest.importorskip('dion.newton_schulz_triton')
    torch.manual_seed(7)
    gradient = torch.randn(shape, device='cuda', dtype=torch.bfloat16)

    expected = _reference_cape(gradient)
    actual = cape_polar(gradient)

    torch.testing.assert_close(actual, expected)
