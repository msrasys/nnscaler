# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import torch

from nnscaler import AsyncResult, ModuleStreams, register_async_op
from nnscaler.runtime.stream import drain_stream_holds

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


class PreparedSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output):
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * 2 * x, None


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('managed', [False, True])
def test_async_submission_precedes_side_compute_and_closes_on_error(fail, managed):
    x = torch.randn(1 << 16, device='cuda', requires_grad=True)
    expected = x.detach().clone().requires_grad_()
    done, side_done = torch.cuda.Event(), torch.cuda.Event()
    communication = torch.cuda.Stream()
    waited = []

    def operation(value):
        return value.square()

    def submit(value):
        caller = torch.cuda.current_stream()
        communication.wait_stream(caller)
        with torch.cuda.stream(communication), torch.no_grad():
            output = torch.empty_like(value)
            torch.cuda._sleep(500_000_000)
            torch.square(value, out=output)
            done.record()
        value.record_stream(communication)
        if managed:
            output.record_stream(caller)

        def wait():
            waited.append(True)
            torch.cuda.current_stream().wait_event(done)

        return AsyncResult(output, wait, manages_lifetime=managed)

    register_async_op(operation, submit=submit, complete=lambda value, original: PreparedSquare.apply(original, value))
    # Cold allocation or CUDA lazy module loading can synchronize away overlap.
    with torch.cuda.stream(communication), torch.no_grad():
        warm_output = torch.empty_like(x)
        torch.square(x, out=warm_output)
        done.record()
    with ModuleStreams() as streams:
        with streams.fork('warm', 'async_op_test', (x,)):
            warm = x.sin()
        streams.publish((warm,))
    del warm, warm_output
    torch.cuda.synchronize()
    try:
        with ModuleStreams() as streams:
            streams.capture('side')
            pending = streams.submit(operation, x)
            with streams.fork('side', 'async_op_test', (x,)):
                side = x.sin()
                side_done.record()
            streams.publish((side,))
            side_done.synchronize()
            assert not done.query(), 'Side computation ran after communication completion'
            if fail:
                raise ValueError('side branch failed')
            result = pending.complete(x)
            streams.join('side')
            result = result + side
    except ValueError as error:
        assert fail and str(error) == 'side branch failed'
    assert waited == [True]
    if not fail:
        reference = expected.square() + expected.sin()
        result.sum().backward()
        reference.sum().backward()
        torch.testing.assert_close(result, reference)
        torch.testing.assert_close(x.grad, expected.grad)
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)


@pytest.mark.parametrize('managed', [False, True])
def test_async_storage_survives_foreign_stream_reuse(managed):
    producer, caller, private = (torch.cuda.Stream() for _ in range(3))
    size = 1 << 20
    for stream in (producer, caller, private):
        with torch.cuda.stream(stream):
            warm = [torch.empty(size, device='cuda').fill_(-11) for _ in range(16)]
            cloned = warm[0].clone()
        del warm, cloned
    torch.cuda.synchronize()
    with torch.cuda.stream(producer):
        inputs = [torch.full((size,), 3., device='cuda')]

    def identity(x):
        return x.clone()

    def submit(x):
        private.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(private):
            torch.cuda._sleep(100_000_000)
            output = x.clone()
            done = torch.cuda.Event()
            done.record()
        if managed:
            x.record_stream(private)
            output.record_stream(caller)
        return AsyncResult(output, lambda: caller.wait_event(done), manages_lifetime=managed)

    register_async_op(identity, submit=submit, complete=lambda output, x: output)
    with torch.cuda.stream(caller), ModuleStreams() as streams:
        caller.wait_stream(producer)
        pending = streams.submit(identity, inputs[0])
        output = pending.complete(inputs.pop())
        torch.cuda._sleep(100_000_000)
        result = output + 7
        del output
    for stream in (producer, private):
        with torch.cuda.stream(stream):
            pressure = [torch.empty(size, device='cuda').fill_(-19) for _ in range(16)]
        del pressure
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.full_like(result, 10), rtol=0, atol=0)
    drain_stream_holds(wait=True)
