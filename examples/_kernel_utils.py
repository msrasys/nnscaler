# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Device-aware optional kernel loading for the model examples."""

import importlib
import inspect
import logging
from functools import lru_cache, partial
from typing import Callable, Optional

import torch


_logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def load_optional_kernel(module: str, name: str) -> Optional[Callable]:
    try:
        kernel = getattr(importlib.import_module(module), name)
        if module.startswith("fla.ops.") and "head_first" in inspect.signature(kernel).parameters:
            kernel = partial(kernel, head_first=False)
    except Exception as exc:
        _logger.warning("Cannot load %s.%s; using the PyTorch reference: %s", module, name, exc)
        return None
    _logger.info("Loaded optional kernel %s.%s", module, name)
    return kernel


def fla_kernel(op: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Optional[Callable]:
    if not q.is_cuda:
        return None
    if not (
        k.is_cuda and v.is_cuda
        and q.device == k.device == v.device
        and q.ndim == k.ndim == v.ndim == 4
        and q.shape == k.shape and q.shape[:2] == v.shape[:2]
        and q.dtype == k.dtype == v.dtype
        and q.dtype in (torch.float16, torch.bfloat16)
        and all(size > 0 for tensor in (q, k, v) for size in tensor.shape)
        and q.shape[-1] <= 256
    ):
        _warn_fla_fallback(op, "requires matching FP16/BF16 CUDA Q/K/V, nonempty inputs and key head dim <= 256")
        return None
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        _warn_fla_fallback(op, "requires an Ampere or newer GPU")
        return None
    return load_optional_kernel(f"fla.ops.{op}", f"chunk_{op}")


@lru_cache(maxsize=None)
def _warn_fla_fallback(op: str, reason: str) -> None:
    _logger.warning("FLA chunk_%s %s; using the PyTorch reference", op, reason)
