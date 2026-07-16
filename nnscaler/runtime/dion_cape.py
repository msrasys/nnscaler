#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""CAPE orthogonalization for Dion's Muon optimizer."""

import torch


try:
    from dion.newton_schulz_triton import newton_schulz_triton
except ImportError as e:
    _DION_TRITON_IMPORT_ERROR = e

    def newton_schulz_triton(*args, **kwargs):
        raise ImportError(
            "DION's newton_schulz_triton is not installed. "
            "Install DION to use the CAPE optimizer."
        ) from _DION_TRITON_IMPORT_ERROR


def _normalize_frobenius(
    X: torch.Tensor, epsilon=1e-7, safety: float = 1.0
) -> torch.Tensor:
    return X / (X.norm(dim=(-2, -1), keepdim=True) * safety + epsilon)


@torch.no_grad()
def mclip(
    X: torch.Tensor, P: torch.Tensor, a: float, epsilon=1e-7
) -> torch.Tensor:
    transposed = X.size(-2) < X.size(-1)
    if transposed:
        X, P = X.mT, P.mT
    aP = a * P
    A = X.mT @ X
    I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).expand_as(A)
    S = newton_schulz_triton(A - a * a * I, epsilon=epsilon)
    X_clipped = 0.5 * (X + aP + (aP - X) @ S)
    if transposed:
        X_clipped = X_clipped.mT
    return X_clipped


@torch.no_grad()
def cape_polar(
    G: torch.Tensor, epsilon=1e-7, c=3.0, iterations: int = 5
) -> torch.Tensor:
    assert G.ndim >= 2
    dtype = G.dtype
    X = G.float()
    r = min(G.shape[-2], G.shape[-1])
    sqrt_r = r ** 0.5
    X = _normalize_frobenius(X, epsilon=epsilon, safety=1.02)
    a = c / sqrt_r
    P = newton_schulz_triton(X, epsilon=epsilon)
    for _ in range(iterations):
        X = mclip(X, P, a, epsilon)
        X = _normalize_frobenius(X, epsilon=epsilon, safety=1.02)

    # Dion's rms_norm LR adjustment assumes the orthogonalized direction has
    # Frobenius norm sqrt(min(m, n)); keep CAPE on the same update scale.
    X = _normalize_frobenius(X, epsilon=epsilon)
    return (1.2 * sqrt_r * X).to(dtype=dtype)


__all__ = ['cape_polar']
