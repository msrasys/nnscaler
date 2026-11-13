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
@pytest.mark.parametrize('backend', ['sm', 'dma_native'])
@pytest.mark.parametrize('direction', ['h2d', 'd2h'])
@pytest.mark.parametrize('size,offset', [(0, 0), (1, 0), (33, 5), (4097, 3), (2**20, 0), (131447, 0)])
def test_sm_copy_offsets_tail_version_and_new_values(size, offset, direction, backend):
    from nnscaler.runtime.activation_sm_copy import copy_d2h, copy_h2d
    copy = copy_h2d if direction == 'h2d' else copy_d2h
    if backend == 'dma_native':
        from nnscaler.runtime.activation_cuda_copy import copy_dma
        def copy(dst, src, blocks):
            # Exercise partial final packets without issuing tiny copies for large inputs.
            quantum = 7 if size < 5000 else 16381
            assert copy_dma(dst, src, quantum) == (size + quantum - 1) // quantum
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


@pytest.mark.parametrize('direction', ['h2d', 'd2h'])
def test_native_dma_selection_submits_whole_storage_once(direction):
    engine = ActivationTransferEngine(CPUOffloadConfig(**{direction + '_backend': 'dma_native'}, copy_chunk_bytes=7),
                                      torch.device('cpu'), Counter(), _Observation())
    src = torch.arange(37, dtype=torch.uint8)
    dst = torch.empty_like(src)
    calls = []
    def copy(destination, source, chunk):
        assert source is src and destination is dst
        calls.append(chunk)
        destination.copy_(source)
        return 6
    engine._native_dma_copy = copy
    engine._copy_into(dst, src, direction)
    assert calls == [7] and torch.equal(dst, src)
    assert engine.stats[direction + '_copy_chunks'] == 6
    assert engine.stats[direction + '_native_dma_batches'] == 1


@cuda
@pytest.mark.parametrize('direction', ['h2d', 'd2h'])
def test_native_dma_keeps_async_allocations_alive(direction):
    from nnscaler.runtime.activation_cuda_copy import copy_dma, prepare_cuda_copy
    prepare_cuda_copy()
    host = torch.full((4096,), 31, dtype=torch.uint8, pin_memory=True)
    gpu = torch.full((4096,), 31, dtype=torch.uint8, device='cuda')
    src, dst = (host, gpu) if direction == 'h2d' else (gpu, host)
    dst.zero_()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(40_000_000)
        assert copy_dma(dst, src, 31) == 133
    del src
    if direction == 'h2d':
        del host
        churn = [torch.full((4096,), 99, dtype=torch.uint8, pin_memory=True) for _ in range(64)]
    else:
        del gpu
        churn = [torch.full((4096,), 99, dtype=torch.uint8, device='cuda') for _ in range(64)]
    stream.synchronize()
    assert torch.all(dst == 31)
    assert len(churn) == 64


@cuda
def test_native_dma_rejects_bad_requests_before_submission():
    from nnscaler.runtime.activation_cuda_copy import copy_dma
    gpu = torch.empty(32, dtype=torch.uint8, device='cuda')
    host = torch.empty(32, dtype=torch.uint8, pin_memory=True)
    with pytest.raises(RuntimeError, match='non-negative'):
        copy_dma(gpu, host, -1)
    with pytest.raises(RuntimeError, match='pinned'):
        copy_dma(gpu, torch.empty(32, dtype=torch.uint8), 1)
    with pytest.raises(RuntimeError, match='sizes'):
        copy_dma(gpu, host[:4], 1)
    with pytest.raises(RuntimeError, match='contiguous'):
        copy_dma(gpu[:16], host[::2], 1)


@cuda
@pytest.mark.parametrize('direction', ['h2d', 'd2h'])
def test_native_dma_keeps_deleted_destination_from_reuse(direction):
    from nnscaler.runtime.activation_cuda_copy import copy_dma, prepare_cuda_copy
    prepare_cuda_copy()
    host = torch.full((4096,), 31, dtype=torch.uint8, pin_memory=True)
    gpu = torch.full((4096,), 31, dtype=torch.uint8, device='cuda')
    src, dst = (host, gpu) if direction == 'h2d' else (gpu, host)
    pointer = dst.data_ptr()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(40_000_000)
        copy_dma(dst, src, 31)
    del dst
    if direction == 'h2d':
        del gpu
        churn = [torch.full((4096,), 99, dtype=torch.uint8, device='cuda') for _ in range(64)]
    else:
        del host
        churn = [torch.full((4096,), 99, dtype=torch.uint8, pin_memory=True) for _ in range(64)]
    assert all(x.data_ptr() != pointer for x in churn)
    stream.synchronize()
    assert all(torch.all(x == 99) for x in churn)


@cuda
def test_native_batch_validates_later_descriptor_before_any_copy():
    from nnscaler.runtime.activation_cuda_copy import prepare_cuda_copy
    module = prepare_cuda_copy()
    src = torch.ones(4096, dtype=torch.uint8, device='cuda')
    host = torch.zeros(4096, dtype=torch.uint8, pin_memory=True)
    with pytest.raises(RuntimeError, match='sizes'):
        module.copy_dma_batch([host, host[:3]], [src, src], [0, 0], 1024)
    torch.cuda.synchronize()
    assert not host.any()


@cuda
def test_native_restore_batch_preserves_distinct_producer_dependencies():
    from nnscaler.runtime.activation_cuda_copy import NativeDMABatch, prepare_cuda_copy
    submit = prepare_cuda_copy().copy_dma_batch
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    hosts = [torch.empty(4096, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    sources = [torch.full((4096,), n, dtype=torch.uint8, device='cuda') for n in (17, 31)]
    destinations = [torch.empty_like(src) for src in sources]
    dependencies = [torch.cuda.Event() for _ in hosts]
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        for host, src, event in zip(hosts, sources, dependencies):
            torch.cuda._sleep(10_000_000)
            host.copy_(src, non_blocking=True)
            event.record()
    batch = NativeDMABatch(submit, 31)
    for dst, host, event in zip(destinations, hosts, dependencies):
        batch.append(dst, host, event)
    with torch.cuda.stream(consumer):
        assert batch.submit() == 266
    del batch, hosts, sources, dependencies
    consumer.synchronize()
    assert torch.all(destinations[0] == 17) and torch.all(destinations[1] == 31)
