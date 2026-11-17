import gc
import weakref

import pytest
import torch

from nnscaler.runtime.deferred import DeferredRegion, LaunchPoint, after_backward, notify_async_launch


def test_launch_point_restores_grad_inside_custom_forward():
    x = torch.tensor([2.], requires_grad=True)
    result = []

    class Anchor(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            assert not torch.is_grad_enabled()
            notify_async_launch()
            return value * 3

        @staticmethod
        def backward(ctx, grad):
            return grad * 3

    with LaunchPoint((lambda: result.append(x.square()),)):
        y = Anchor.apply(x)
    (y + result[0]).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([7.]))
    assert not notify_async_launch()
    with pytest.raises(RuntimeError, match='did not call'):
        with LaunchPoint((lambda: None,)):
            pass


def test_deferred_gradients_control_order_and_retention():
    torch.manual_seed(7)
    x = torch.randn(4, 8, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(8, 8))
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    calls = []

    class Parent:
        w = weight

    def compute(module, value):
        return (torch.sin(value @ module.w),)

    state = DeferredRegion(compute, Parent(), ['w'], [((4, 8), x.dtype, True)])
    y, = state.prepare((x,), (weight,))
    assert not state.cleared
    y.register_hook(lambda grad: calls.append('side'))
    dependent = after_backward(x * 2, (y,))

    class Anchor(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            notify_async_launch()
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            calls.append('anchor')
            return grad

    with LaunchPoint((state.launch,)):
        main = Anchor.apply(dependent)
    expected = torch.sin(reference_x @ reference_weight)
    torch.testing.assert_close(y, expected)
    loss = (y + main).sum()
    expected_loss = (expected + reference_x * 2).sum()
    for retain in (True, False):
        calls.clear()
        loss.backward(retain_graph=retain)
        expected_loss.backward(retain_graph=retain)
        torch.testing.assert_close(x.grad, reference_x.grad)
        torch.testing.assert_close(weight.grad, reference_weight.grad)
        assert calls.index('anchor') < calls.index('side')
        assert state.cleared is not retain
        x.grad = weight.grad = reference_x.grad = reference_weight.grad = None
    # Retaining loss/y after backward must not retain inner activations.
    assert state.inner_outputs == ()
    with pytest.raises(RuntimeError, match='second time'):
        loss.backward()


def test_discarded_deferred_graph_releases_inputs():
    x = torch.randn(2, requires_grad=True)
    state = DeferredRegion(lambda parent, value: (value.square(),), None, [], [((2,), x.dtype, True)])
    out, = state.prepare((x,), ())
    state.launch()
    value = weakref.ref(state.values[0])
    del out
    gc.collect()
    assert state.cleared and value() is None


def test_deferred_rejects_observable_aliases():
    x = torch.randn(2, requires_grad=True)
    state = DeferredRegion(lambda parent, value: (value,), None, [], [((2,), x.dtype, True)])
    out, = state.prepare((x,), ())
    with pytest.raises(ValueError, match='alias'):
        state.launch()


def test_original_tensor_hooks_are_not_run_by_the_inner_graph():
    x = torch.randn(4, requires_grad=True)
    calls = []
    x.register_hook(lambda grad: (calls.append(True), grad * 3)[1])
    state = DeferredRegion(lambda parent, value: (value.square(),), None, [], [((4,), x.dtype, True)])
    y, = state.prepare((x,), ())
    state.launch()
    y.sum().backward()
    assert len(calls) == 1
    torch.testing.assert_close(x.grad, 6 * x.detach())


def test_deferred_no_grad_does_not_wait_for_a_backward():
    x = torch.randn(4, requires_grad=True)
    state = DeferredRegion(lambda parent, value: (value.square(),), None, [], [((4,), x.dtype, True)])
    with torch.no_grad():
        y, = state.prepare((x,), ())
        state.launch()
    torch.testing.assert_close(y, x.detach().square())
    assert not y.requires_grad and state.cleared


@pytest.mark.skipif(torch.__version__ < (2, 10), reason='grad_dtype requires PyTorch 2.10+')
def test_deferred_preserves_explicit_gradient_dtype():
    x = torch.randn(4, dtype=torch.bfloat16, requires_grad=True)
    x.grad_dtype = torch.float32
    state = DeferredRegion(lambda parent, value: (value.square(),), None, [], [((4,), x.dtype, True)])
    y, = state.prepare((x,), ())
    state.launch()
    y.sum().backward()
    assert x.grad.dtype == torch.float32
    torch.testing.assert_close(x.grad, (2 * x.detach()).float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_deferred_backward_runs_while_anchor_gradient_is_pending():
    from nnscaler.runtime.stream import ModuleStreams, _get_module_stream, drain_stream_holds
    torch.manual_seed(8)
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(64, 64, device='cuda'))
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    side = _get_module_stream(x.device, 'launch_test')
    comm = torch.cuda.Stream()
    finished = torch.cuda.Event()
    check = False

    class Matmul(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, w):
            ctx.save_for_backward(value, w)
            return value @ w

        @staticmethod
        def backward(ctx, grad):
            value, w = ctx.saved_tensors
            dx, dw = grad @ w.t(), value.t() @ grad
            if check:
                done = torch.cuda.Event()
                done.record()
                done.synchronize()
                assert not finished.query(), 'side backward inherited the communication wait'
            return dx, dw

    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm_x = x.detach().clone().requires_grad_()
        warm_w = weight.detach().clone().requires_grad_()
        Matmul.apply(warm_x, warm_w).sum().backward()
    warm = torch.ones((), device='cuda').expand_as(x) * 3
    torch.cuda.synchronize()
    del warm, warm_x, warm_w

    class Anchor(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            notify_async_launch()
            return value * 3

        @staticmethod
        def backward(ctx, grad):
            caller = torch.cuda.current_stream()
            comm.wait_stream(caller)
            with torch.cuda.stream(comm):
                torch.cuda._sleep(1_000_000_000)
                finished.record()
            caller.wait_event(finished)
            return grad * 3

    class Parent:
        w = weight

    def compute(module, value):
        return (Matmul.apply(value, module.w),)

    with ModuleStreams() as streams:
        y, = streams.prepare('region', 'launch_test', compute, Parent(), (x,), {'w': weight},
                             (((64, 64), x.dtype, True),))
        controlled = after_backward(x, (y,))
        with streams.launch_point(('region',)):
            routed = Anchor.apply(controlled)
        streams.publish((y,))
        streams.join('region')
        output = routed + y
    check = True
    output.sum().backward()
    (reference_x * 3 + reference_x @ reference_weight).sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(weight.grad, reference_weight.grad)
    drain_stream_holds(wait=True)


def test_two_deferred_regions_can_launch_in_dependency_order_at_one_point():
    x = torch.randn(4, requires_grad=True)
    first = DeferredRegion(lambda parent, value: (value.square(),), None, [], [((4,), x.dtype, True)])
    a, = first.prepare((x,), ())
    second = DeferredRegion(lambda parent, value: (value.sin(),), None, [], [((4,), x.dtype, True)])
    b, = second.prepare((a,), ())
    with LaunchPoint((first.launch, second.launch)):
        notify_async_launch()
    torch.testing.assert_close(b, x.square().sin())
    b.sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach() * x.detach().square().cos())
    assert first.cleared and second.cleared
