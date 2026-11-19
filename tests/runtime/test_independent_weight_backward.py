import gc
import weakref

import pytest
import torch

from nnscaler import prepare_async_input, prepare_weight_backward
from nnscaler.runtime.weight_backward import WeightBackwardRegion
from .test_weight_backward import Anchor, Linear


def run_independent(x, w, order):
    region = WeightBackwardRegion("independent_weights", (w,), independent=True)
    with region.anchor():
        routed = Anchor.apply(prepare_async_input(x), order)
    region.capture_output(routed)
    with region.compute():
        (value,) = region.order_inputs((x,))
        binding = prepare_weight_backward((w,))
        shared = Linear.apply(value, binding.parameters[0], binding, order)
    # Give the anchor higher ready-queue priority than the independent branch.
    # Its registration dependency must still stop it from submitting too early.
    routed.grad_fn._set_sequence_nr((1 << 60) + routed.grad_fn._sequence_nr())
    result = routed.sin() + shared
    region.close_forward()
    return result, region


def test_independent_registration_precedes_anchor_even_with_anchor_priority():
    torch.manual_seed(7)
    x = torch.randn(8, 4, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(4, 4))
    rx = x.detach().clone().requires_grad_()
    rw = w.detach().clone().requires_grad_()
    order = []
    actual, region = run_independent(x, w, order)
    expected = (rx * 2).sin() + rx @ rw
    for iteration in range(2):
        actual.square().sum().backward(retain_graph=iteration == 0)
        expected.square().sum().backward(retain_graph=iteration == 0)
        for a, b in ((actual, expected), (x.grad, rx.grad), (w.grad, rw.grad)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert not region.tasks
    assert order == ["dx", "submit", "dw"] * 2


def test_registration_token_does_not_retain_anchor_output_storage():
    x = torch.randn(8, 4, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(4, 4))
    region = WeightBackwardRegion("token", (w,), independent=True)
    with region.anchor():
        output = Anchor.apply(prepare_async_input(x), [])
    output_ref = weakref.ref(output)
    region.capture_output(output)
    assert region.registration_token.untyped_storage().nbytes() == 0
    del output
    gc.collect()
    assert output_ref() is None
    a, b, nested = region.order_inputs((x, x, {"x": x}))
    assert a is b and a is nested["x"]
    assert region.registration_token is None
    with pytest.raises(RuntimeError, match="unused anchor"):
        region.order_inputs((x,))
    region.close_forward(failed=True)
    ref = weakref.ref(region)
    del a, b, nested, region
    gc.collect()
    assert ref() is None


def test_missing_registration_and_nondifferentiable_inputs_are_rejected():
    x = torch.randn(8, 4, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(4, 4))
    region = WeightBackwardRegion("missing", (w,), independent=True)
    with pytest.raises(RuntimeError, match="unused anchor"):
        region.order_inputs((x,))
    with region.anchor():
        output = Anchor.apply(prepare_async_input(x), [])
    region.capture_output(output)
    with pytest.raises(ValueError, match="differentiable activation"):
        region.order_inputs((x.detach(),))
    region.close_forward(failed=True)


def test_pruned_parameter_edges_and_abandoned_forward_release_state():
    x = torch.randn(8, 4, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(4, 4))
    out, region = run_independent(x, w, [])
    (dx,) = torch.autograd.grad(out.sum(), (x,))
    expected = 2 * (2 * x).cos() + torch.ones_like(out) @ w.t()
    torch.testing.assert_close(dx, expected, rtol=0, atol=0)
    assert w.grad is None and not region.tasks
    ref = weakref.ref(region)
    del out, region
    gc.collect()
    assert ref() is None
    out, region = run_independent(x, w, [])
    ref = weakref.ref(region)
    del out, region
    gc.collect()
    assert ref() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_independent_cuda_readiness_overlap_and_no_early_input_join():
    from nnscaler import capture_async_backward
    from nnscaler.runtime.stream import _get_module_stream, drain_stream_holds

    torch.cuda._sleep(1)
    x = torch.randn(256, 64, device="cuda", requires_grad=True)
    w = torch.nn.Parameter(torch.randn(64, 64, device="cuda"))
    rx = x.detach().clone().requires_grad_()
    rw = w.detach().clone().requires_grad_()
    ((rx * 2).sin() + rx @ rw).sum().backward()
    expected_x, expected_w = rx.grad.clone(), rw.grad.clone()
    hold = x.expand_as(x)  # Keep the original input accumulator on the caller.
    region = WeightBackwardRegion("independent_cuda_weights", (w,), independent=True)
    side = _get_module_stream(x.device, "independent_cuda_compute")
    comm = torch.cuda.Stream()
    ones = torch.ones_like(x)
    for stream in (side, region.stream):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x.detach() @ w.detach()
            ones @ w.detach().t()
            x.detach().t() @ ones
    torch.cuda.synchronize()
    stamps = {}
    order = []

    def stamp(name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        stamps[name] = event

    class Communication(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.point = capture_async_backward()
            return value * 2

        @staticmethod
        def backward(ctx, grad):
            assert order == ["registered"]
            ctx.point.record_ready()
            comm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(comm):
                torch.cuda._sleep(25_000_000)
                stamp("comm_done")
            ctx.point.submitted()
            torch.cuda.current_stream().wait_event(stamps["comm_done"])
            return grad * 2

    class Branch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value, parameter, binding):
            ctx.save_for_backward(value, parameter)
            ctx.binding = binding
            return value @ parameter

        @staticmethod
        def backward(ctx, grad):
            value, parameter = ctx.saved_tensors
            torch.cuda._sleep(2_000_000)
            dx = grad @ parameter.t()
            stamp("dx_ready")

            def compute_weight():
                stamp("weight_started")
                torch.cuda._sleep(40_000_000)
                result = value.t() @ grad
                stamp("weight_done")
                return (result,)

            ctx.binding.defer(compute_weight, (value, grad))
            order.append("registered")
            return dx, None, None

    with region.anchor():
        routed = Communication.apply(prepare_async_input(x))
    region.capture_output(routed)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), region.compute():
        (value,) = region.order_inputs((x,))
        binding = prepare_weight_backward((w,))
        shared = Branch.apply(value, binding.parameters[0], binding)
    torch.cuda.current_stream().wait_stream(side)
    result = routed.sin() + shared
    region.close_forward()
    x.register_hook(lambda gradient: (stamp("upstream_ready"), gradient)[1])
    result.sum().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(x.grad, expected_x, rtol=0, atol=0)
    torch.testing.assert_close(w.grad, expected_w, rtol=0, atol=0)
    assert stamps["dx_ready"].elapsed_time(stamps["weight_started"]) >= 0
    assert stamps["weight_started"].elapsed_time(stamps["comm_done"]) > 0
    assert stamps["upstream_ready"].elapsed_time(stamps["weight_done"]) > 0
    assert not region.tasks
    del hold
    drain_stream_holds(wait=True)
