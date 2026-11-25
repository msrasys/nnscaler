# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import torch

from nnscaler import ComputeConfig
from nnscaler.runtime import stream


def test_queue_waits_only_oldest_boundary_and_keeps_failed_work(monkeypatch):
    streams = [object(), object()]
    current = [streams[0]]
    recorded, waited = [], []

    class Event:
        def record(self, caller):
            recorded.append((self, caller))

        def synchronize(self):
            waited.append(self)

    monkeypatch.setattr(stream, '_cuda_task_queues', {})
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: current[0])
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    for _ in range(2):
        with stream.cuda_task_queue(2):
            assert not waited
    current[0] = streams[1]
    with stream.cuda_task_queue(2):
        assert not waited
    current[0] = streams[0]
    with pytest.raises(RuntimeError, match='task failed'):
        with stream.cuda_task_queue(2):
            assert waited == [recorded[0][0]]
            raise RuntimeError('task failed')
    assert len(recorded) == 4
    stream.drain_cuda_tasks()
    assert set(waited) == {event for event, _ in recorded}


def test_queue_configuration_is_serialized_and_validated(tmp_path, monkeypatch):
    monkeypatch.setenv('NNSCALER_MAX_PENDING_BACKWARD_TASKS', '2')
    configured = ComputeConfig(1, 1)
    assert configured.max_pending_backward_tasks == 2
    path = tmp_path / 'config.pt'
    ComputeConfig.safe_dump_to_file(configured, path)
    monkeypatch.setenv('NNSCALER_MAX_PENDING_BACKWARD_TASKS', '1')
    assert ComputeConfig.safe_load_from_file(path) == configured
    payload = torch.load(path, weights_only=False)
    payload.pop('max_pending_backward_tasks')
    torch.save(payload, path)
    assert ComputeConfig.safe_load_from_file(path).max_pending_backward_tasks == 0
    assert not ComputeConfig.safe_equals(configured, ComputeConfig(1, 1))
    for limit in (-1, True, 1.5):
        with pytest.raises(ValueError, match='nonnegative integer'):
            ComputeConfig(1, 1, max_pending_backward_tasks=limit)
        with pytest.raises(ValueError, match='nonnegative integer'):
            with stream.cuda_task_queue(limit):
                pass
    with pytest.raises(ValueError, match='complete backward tasks'):
        ComputeConfig(1, 1, use_end2end=True, use_fbw=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_backward_boundary_includes_side_stream_without_serializing_first_task(monkeypatch):
    monkeypatch.setattr(stream, '_cuda_task_queues', {})
    caller, side = torch.cuda.current_stream(), torch.cuda.Stream()
    finished = torch.cuda.Event()

    class Delayed(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.square()

        @staticmethod
        def backward(ctx, grad):
            x, = ctx.saved_tensors
            torch.cuda._sleep(150_000_000)
            result = grad * 2 * x
            finished.record()
            return result

    x = torch.randn(128, device='cuda', requires_grad=True)
    # Warm both initial gradient allocation and subsequent accumulation kernels.
    for _ in range(3):
        side.wait_stream(caller)
        with torch.cuda.stream(side):
            warm = Delayed.apply(x)
        caller.wait_stream(side)
        warm.record_stream(caller)
        warm.sum().backward()
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    x.grad.zero_()
    side.wait_stream(caller)
    with torch.cuda.stream(side):
        output = Delayed.apply(x)
    caller.wait_stream(side)
    output.record_stream(caller)
    torch.cuda.synchronize()
    try:
        with stream.cuda_task_queue(1):
            output.sum().backward()
        assert not finished.query(), 'The first submitted task must remain asynchronous'
        with stream.cuda_task_queue(1):
            assert finished.query(), 'The next task must wait for the previous side backward'
            torch.testing.assert_close(x.grad, 2*x, rtol=0, atol=0)
    finally:
        stream.drain_cuda_tasks()
