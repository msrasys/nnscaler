# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
from collections import Counter

import pytest
import torch

from nnscaler.runtime.activation_config import CPUOffloadConfig
from nnscaler.runtime.activation_observer import _Observation
from nnscaler.runtime.activation_transfer import ActivationTransferEngine


@pytest.mark.parametrize('values', [dict(h2d_backend='other'), dict(h2d_sm_blocks=0),
                                     dict(h2d_sm_blocks=True), dict(h2d_sm_blocks=-1)])
def test_invalid_sm_restore_settings(values):
    with pytest.raises(ValueError):
        CPUOffloadConfig(**values)


def test_sm_selection_is_restore_only_and_preserves_chunk_offsets():
    engine = ActivationTransferEngine(CPUOffloadConfig(h2d_backend='sm', copy_chunk_bytes=7),
                                      torch.device('cpu'), Counter(), _Observation())
    calls = []
    def copy(dst, src, blocks):
        calls.append((src.numel(), blocks))
        dst.copy_(src)
    engine._sm_copy = copy
    src = torch.arange(37, dtype=torch.uint8)
    dst = torch.empty_like(src)
    engine._copy_into(dst, src, 'd2h')
    assert not calls and torch.equal(dst, src)
    engine._copy_into(dst, src, 'h2d')
    assert calls == [(7, 16)] * 5 + [(2, 16)]
    assert engine.stats['h2d_sm_copy_chunks'] == 6 and torch.equal(dst, src)


cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                          reason='NVIDIA CUDA required')


@cuda
@pytest.mark.parametrize('size,offset', [(0, 0), (1, 0), (33, 5), (4097, 3), (2**20, 0)])
def test_sm_copy_offsets_tail_version_and_new_host_values(size, offset):
    from nnscaler.runtime.activation_sm_copy import copy_h2d
    host = torch.empty(size + offset + 7, dtype=torch.uint8, pin_memory=True)
    backing = torch.full((size + offset + 7,), 251, device='cuda', dtype=torch.uint8)
    src, dst = host[offset:offset+size], backing[offset:offset+size]
    for seed in [11, 73]:
        src.fill_(seed)
        version = dst._version
        copy_h2d(dst, src, 16)
        torch.cuda.synchronize()
        assert dst._version == version + 1
        assert torch.equal(dst.cpu(), src)
        assert torch.all(backing[:offset] == 251)
        assert torch.all(backing[offset+size:] == 251)


@cuda
def test_sm_copy_keeps_pinned_source_alive_during_async_read():
    from nnscaler.runtime.activation_sm_copy import copy_h2d, prepare_sm_h2d
    prepare_sm_h2d()
    stream = torch.cuda.Stream()
    dst = torch.empty(4096, device='cuda', dtype=torch.uint8)
    src = torch.full((4096,), 17, dtype=torch.uint8, pin_memory=True)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(40_000_000)
        copy_h2d(dst, src, 4)
    del src
    # Freed pinned blocks must not be reused while the kernel is still queued.
    churn = [torch.full((4096,), 99, dtype=torch.uint8, pin_memory=True) for _ in range(64)]
    stream.synchronize()
    assert torch.all(dst == 17)
    assert len(churn) == 64


@cuda
def test_sm_copy_rejects_unpinned_or_nonbyte_sources():
    from nnscaler.runtime.activation_sm_copy import copy_h2d
    dst = torch.empty(32, device='cuda', dtype=torch.uint8)
    with pytest.raises(RuntimeError, match='pinned'):
        copy_h2d(dst, torch.ones(32, dtype=torch.uint8), 16)
    with pytest.raises(RuntimeError, match='uint8'):
        copy_h2d(dst, torch.ones(32, dtype=torch.float32, pin_memory=True), 16)
