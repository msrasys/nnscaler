# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
from collections import Counter

import pytest
import torch

from nnscaler.runtime.activation_config import CPUOffloadConfig
from nnscaler.runtime.activation_observer import _Observation
from nnscaler.runtime.activation_transfer import ActivationTransferEngine


@pytest.mark.parametrize('values', [dict(h2d_backend='other'), dict(h2d_sm_blocks=0),
                                     dict(h2d_sm_blocks=True), dict(h2d_sm_blocks=-1),
                                     dict(d2h_backend='other'), dict(d2h_sm_blocks=0),
                                     dict(d2h_sm_blocks=True), dict(d2h_sm_blocks=-1)])
def test_invalid_sm_transfer_settings(values):
    with pytest.raises(ValueError):
        CPUOffloadConfig(**values)


@pytest.mark.parametrize('direction,blocks', [('h2d', 16), ('d2h', 4)])
def test_sm_selection_is_directional_and_preserves_chunk_offsets(direction, blocks):
    engine = ActivationTransferEngine(CPUOffloadConfig(**{direction + '_backend': 'sm'}, copy_chunk_bytes=7),
                                      torch.device('cpu'), Counter(), _Observation())
    calls = []
    def copy(dst, src, blocks):
        calls.append((src.numel(), blocks))
        dst.copy_(src)
    engine._sm_copies[direction] = copy
    src = torch.arange(37, dtype=torch.uint8)
    dst = torch.empty_like(src)
    engine._copy_into(dst, src, 'd2h' if direction == 'h2d' else 'h2d')
    assert not calls and torch.equal(dst, src)
    engine._copy_into(dst, src, direction)
    assert calls == [(7, blocks)] * 5 + [(2, blocks)]
    assert engine.stats[direction + '_sm_copy_chunks'] == 6 and torch.equal(dst, src)


cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                          reason='NVIDIA CUDA required')


@cuda
@pytest.mark.parametrize('direction', ['h2d', 'd2h'])
@pytest.mark.parametrize('size,offset', [(0, 0), (1, 0), (33, 5), (4097, 3), (2**20, 0), (131447, 0)])
def test_sm_copy_offsets_tail_version_and_new_values(size, offset, direction):
    from nnscaler.runtime.activation_sm_copy import copy_d2h, copy_h2d
    copy = copy_h2d if direction == 'h2d' else copy_d2h
    host = torch.full((size + offset + 7,), 251, dtype=torch.uint8, pin_memory=True)
    backing = torch.full((size + offset + 7,), 251, device='cuda', dtype=torch.uint8)
    src, dst = host[offset:offset+size], backing[offset:offset+size]
    target = backing
    if direction == 'd2h':
        src, dst, target = dst, src, host
    for seed in [11, 73]:
        src.fill_(seed)
        version = dst._version
        copy(dst, src, 4)
        torch.cuda.synchronize()
        assert dst._version == version + 1
        assert torch.equal(dst.cpu(), src.cpu())
        assert torch.all(target[:offset] == 251)
        assert torch.all(target[offset+size:] == 251)


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


@cuda
def test_sm_d2h_keeps_gpu_source_alive_during_async_read():
    from nnscaler.runtime.activation_sm_copy import copy_d2h, prepare_sm_copy
    prepare_sm_copy()
    stream = torch.cuda.Stream()
    src = torch.full((4096,), 17, device='cuda', dtype=torch.uint8)
    dst = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(40_000_000)
        copy_d2h(dst, src, 4)
    del src
    churn = [torch.full((4096,), 99, device='cuda', dtype=torch.uint8) for _ in range(64)]
    stream.synchronize()
    assert torch.all(dst == 17)
    assert len(churn) == 64


@cuda
def test_sm_d2h_keeps_pinned_destination_alive_during_async_write():
    from nnscaler.runtime.activation_sm_copy import copy_d2h, prepare_sm_copy
    prepare_sm_copy()
    stream = torch.cuda.Stream()
    src = torch.full((4096,), 17, device='cuda', dtype=torch.uint8)
    dst = torch.empty(4096, dtype=torch.uint8, pin_memory=True)
    pointer = dst.data_ptr()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(40_000_000)
        copy_d2h(dst, src, 4)
    del dst
    churn = [torch.full((4096,), 99, dtype=torch.uint8, pin_memory=True) for _ in range(64)]
    assert all(x.data_ptr() != pointer for x in churn)
    stream.synchronize()
    assert all(torch.all(x == 99) for x in churn)


@cuda
def test_sm_d2h_rejects_unpinned_or_nonbyte_destinations():
    from nnscaler.runtime.activation_sm_copy import copy_d2h
    src = torch.ones(32, device='cuda', dtype=torch.uint8)
    with pytest.raises(RuntimeError, match='pinned'):
        copy_d2h(torch.empty(32, dtype=torch.uint8), src, 4)
    with pytest.raises(RuntimeError, match='uint8'):
        copy_d2h(torch.empty(32, dtype=torch.float32, pin_memory=True), src, 4)
