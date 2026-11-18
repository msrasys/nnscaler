import pytest
import torch

from nnscaler.runtime.deferred import (
    PrequeuedLaunchPoint, prepare_async_input, notify_async_launch,
    native_async_launch_pending,
)


def test_prepare_binds_existing_graph_without_releasing_work():
    x = torch.randn(7, requires_grad=True)
    expected_x = x.detach().clone().requires_grad_()
    side = x.square()  # Host graph has already been built.
    calls = []
    with PrequeuedLaunchPoint((lambda: calls.append('released'),),
                              lambda ready: calls.append('ready'), lambda: (side,)):
        routed = prepare_async_input(x)
        assert calls == ['ready'] and native_async_launch_pending()
        notify_async_launch()
        assert calls == ['ready', 'released']
        assert not native_async_launch_pending()
        assert notify_async_launch() is False
    (side + routed.sin()).sum().backward()
    (expected_x.square() + expected_x.sin()).sum().backward()
    torch.testing.assert_close(x.grad, expected_x.grad, rtol=0, atol=0)


def test_missing_or_duplicate_protocol_calls_rejected():
    x = torch.ones(3)
    with pytest.raises(RuntimeError, match='prepare its input'):
        with PrequeuedLaunchPoint((), lambda _: None, lambda: ()):
            notify_async_launch()
    with pytest.raises(RuntimeError, match='exactly one'):
        with PrequeuedLaunchPoint((), lambda _: None, lambda: ()):
            prepare_async_input(x)
            prepare_async_input(x)
    with pytest.raises(RuntimeError, match='notify_async_launch'):
        with PrequeuedLaunchPoint((), lambda _: None, lambda: ()):
            prepare_async_input(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_prequeued_module_waits_until_submission_and_preserves_gradients():
    from nnscaler.runtime.stream import ModuleStreams, drain_stream_holds, _get_module_stream
    from nnscaler import capture_async_backward
    torch.cuda._sleep(1)
    x = torch.randn(512, 64, device='cuda', requires_grad=True)
    w = torch.nn.Parameter(torch.randn(64, 64, device='cuda'))
    rx = x.detach().clone().requires_grad_()
    rw = w.detach().clone().requires_grad_()
    comm = torch.cuda.Stream()
    seen = []
    started = torch.cuda.Event(enable_timing=True)
    done = torch.cuda.Event(enable_timing=True)
    class Parent:
        weight = w
    def compute(parent, value):
        seen.append('built')
        started.record()
        return ((value @ parent.weight).sin(),)
    # Warm all kernels before the first gate, including backward.
    ((x @ w).sin().sum()).backward()
    x.grad = w.grad = None
    torch.cuda.synchronize()
    side = _get_module_stream(x.device, 'prequeue_test')
    with torch.cuda.stream(side):
        ((x.detach() @ w.detach()).sin()).sum()
    torch.cuda.synchronize()
    class Anchor(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.point = capture_async_backward()
            comm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(comm):
                torch.cuda._sleep(15_000_000)
                done.record()
            notify_async_launch()
            torch.cuda.current_stream().wait_event(done)
            return value * 3
        @staticmethod
        def backward(ctx, grad):
            ctx.point.record_ready()
            return grad * 3
    with ModuleStreams() as scope:
        scope.schedule_native('work', 'prequeue_test', compute, Parent(), (x,), {'weight': w},
                              prequeue=True, preserve_accumulators=True)
        assert seen == ['built']
        assert not started.query()
        with scope.launch_point(('work',), native=True, prequeued=True, backward_keys=('work',)):
            routed = Anchor.apply(prepare_async_input(x))
        y, = scope.native_result('work')
        scope.join('work')
        gates = tuple(scope.gates)
        output = routed + y
    output.sum().backward()
    (3 * rx + (rx @ rw).sin()).sum().backward()
    torch.cuda.synchronize()
    assert started.elapsed_time(done) > 0
    torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
    torch.testing.assert_close(w.grad, rw.grad, rtol=0, atol=0)
    assert gates and all(not gate.fallback() for gate in gates)
    drain_stream_holds(wait=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_exception_opens_gate_before_segment_exit():
    from nnscaler.runtime.stream import ModuleStreams, drain_stream_holds
    x = torch.zeros(7, device='cuda')
    # Warm the elementwise operation before gating.
    x.add_(2); x.zero_(); torch.cuda.synchronize()
    def compute(_, value):
        return (value + 2,)
    with pytest.raises(ValueError, match='injected'):
        with ModuleStreams() as scope:
            scope.schedule_native('work', 'prequeue_failure', compute, None, (x,), {}, prequeue=True)
            raise ValueError('injected')
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)
