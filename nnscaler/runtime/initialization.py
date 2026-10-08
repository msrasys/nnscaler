#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""In-memory initialization of generated modules."""

from contextlib import contextmanager, nullcontext
import random
from typing import TYPE_CHECKING, Callable, Dict, Optional, Type

import numpy as np
import torch

if TYPE_CHECKING:
    from nnscaler.runtime.module import AttrMeta


@contextmanager
def preserve_rng_state():
    """Preserve Python, NumPy, torch CPU and current-device CUDA RNG states."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def _initialization_rng(seed: Optional[int]):
    with preserve_rng_state() if seed is not None else nullcontext():
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            # torch.manual_seed also changes other CUDA devices, outside the saved RNG scope.
            torch.random.default_generator.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        yield


def create_init_module(
    module_class: Type[torch.nn.Module],
    module_fn: Optional[Callable[[], torch.nn.Module]],
    module_dtype: Optional[torch.dtype],
    *,
    seed: Optional[int] = None,
) -> torch.nn.Module:
    with _initialization_rng(seed):
        module = module_class() if module_fn is None else module_fn()
        if type(module) is not module_class:
            raise ValueError(f"module_fn should return a {module_class} instance.")
        if module_dtype is not None:
            module.to(dtype=module_dtype)
    return module


def create_partial_init_weights(
    module_class: Type[torch.nn.Module],
    attr_meta_map: Dict[str, "AttrMeta"],
    *,
    seed: int,
) -> Dict[str, torch.Tensor]:
    initializer = getattr(module_class, '__partial__init__', None)
    if not callable(initializer):
        raise RuntimeError(
            "custom initialization requires the original module to define "
            "__partial__init__(attr_meta_map) as a staticmethod or classmethod."
        )
    with _initialization_rng(seed):
        return initializer(attr_meta_map)
