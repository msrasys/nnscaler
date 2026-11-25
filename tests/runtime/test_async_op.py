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
def test_async_submission_precedes_side_compute_and_closes_on_error(fail):
    x = torch.randn(1 << 16, device='cuda', requires_grad=True)
    expected = x.detach().clone().requires_grad_()
    done, side_done = torch.cuda.Event(), torch.cuda.Event()
    communication = torch.cuda.Stream()
    waited = []

    def operation(value):
        return value.square()

    def submit(value):
        communication.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(communication), torch.no_grad():
            output = torch.empty_like(value)
            torch.cuda._sleep(500_000_000)
            torch.square(value, out=output)
            done.record()
        value.record_stream(communication)

        def wait():
            waited.append(True)
            torch.cuda.current_stream().wait_event(done)

        return AsyncResult(output, wait)

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
