# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Optional SM-based restore transport; storage ownership stays with the store."""
from functools import lru_cache
from pathlib import Path
import threading

import torch

_build_lock = threading.Lock()


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load
    source = Path(__file__).with_name('csrc')
    return load(
        name='nnscaler_activation_sm_copy_v1',
        sources=[str(source / 'activation_sm_copy.cpp'), str(source / 'activation_sm_copy.cu')],
        extra_cflags=['-O3', '-Wno-deprecated-declarations'],
        extra_cuda_cflags=['-O3'], verbose=False,
    )


def prepare_sm_h2d():
    """Build/load once, before entering distributed communication or graph capture."""
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise RuntimeError('SM activation restore requires CUDA')
    # lru_cache alone can invoke a builder concurrently on different threads.
    with _build_lock:
        return _extension()


def copy_h2d(destination, source, blocks):
    """Copy contiguous raw bytes on the caller's current CUDA stream.

    The native implementation records both allocations with their PyTorch
    allocators. Producer readiness and consumer events are supplied by the
    ordinary transfer engine; this function introduces no host wait.
    """
    module = prepare_sm_h2d()
    module.copy_h2d(destination, source, blocks)
    torch.autograd.graph.increment_version(destination)
