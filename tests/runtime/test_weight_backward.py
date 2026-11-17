import weakref

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nnscaler import capture_async_backward, prepare_async_input, prepare_weight_backward
from nnscaler.runtime.weight_backward import WeightBackwardRegion


class Anchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, order):
        ctx.point = capture_async_backward()
        ctx.order = order
        return x * 2

    @staticmethod
    def backward(ctx, dy):
        ctx.order.append('submit')
        if ctx.point is not None:
            ctx.point.record_ready()
            ctx.point.submitted()
        return dy * 2, None


class Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, binding, order):
        ctx.save_for_backward(x, w)
        ctx.binding, ctx.order = binding, order
        return x @ w

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        dx = dy @ w.t()
        ctx.order.append('dx')
        if ctx.binding is None:
            return dx, x.float().t() @ dy.float(), None, None
        order = ctx.order

        def weight(x=x, dy=dy):
            order.append('dw')
            return (x.float().t() @ dy.float(),)

        ctx.binding.defer(weight, (x, dy))
        return dx, None, None, None


def run(x, w, order, recompute=False):
    region = WeightBackwardRegion('weights', (w,))
    with region.anchor():
        routed = Anchor.apply(prepare_async_input(x), order)

    def compute(value):
        with region.compute():
            binding = prepare_weight_backward((w,))
            return Linear.apply(value, binding.parameters[0], binding, order)

    output = checkpoint(compute, routed, use_reentrant=False) if recompute else compute(routed)
    region.close_forward()
    return output, region


@pytest.mark.parametrize('recompute', [False, True])
def test_local_weight_gradients_preserve_autograd_and_repeated_backward(recompute):
    torch.manual_seed(7)
    x = torch.randn(8, 4, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(4, 6))
    rx, rw = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    order = []
    hooks = []
    w.register_post_accumulate_grad_hook(lambda p: hooks.append(p.grad.clone()))
    out, region = run(x, w, order, recompute)
    ref = (rx * 2) @ rw
    for iteration in range(2):
        out.square().sum().backward(retain_graph=iteration == 0)
        ref.square().sum().backward(retain_graph=iteration == 0)
        torch.testing.assert_close(out, ref)
        torch.testing.assert_close(x.grad, rx.grad)
        torch.testing.assert_close(w.grad, rw.grad)
        assert not region.tasks
    assert order == ['dx', 'submit', 'dw'] * 2
    assert len(hooks) == 2


def test_live_forwards_and_tied_parameters_do_not_share_weight_tasks():
    w = torch.nn.Parameter(torch.randn(4, 6))
    rw = w.detach().clone().requires_grad_()
    xs = [torch.randn(8, 4, requires_grad=True) for _ in range(2)]
    rxs = [x.detach().clone().requires_grad_() for x in xs]
    calls = [run(x, w, []) for x in xs]
    sum(y.square().sum() for y, _ in calls).backward()
    sum(((x * 2) @ rw).square().sum() for x in rxs).backward()
    torch.testing.assert_close(w.grad, rw.grad)
    for x, rx in zip(xs, rxs):
        torch.testing.assert_close(x.grad, rx.grad)
    assert all(not region.tasks for _, region in calls)


def test_pruned_parameter_edges_clean_tasks_and_no_python_reference_cycle():
    x = torch.randn(8, 4, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(4, 6))
    out, region = run(x, w, [])
    dx, = torch.autograd.grad(out.sum(), (x,))
    torch.testing.assert_close(dx, 2 * torch.ones_like(out) @ w.t())
    assert w.grad is None and not region.tasks
    reference = weakref.ref(region)
    del out, region
    assert reference() is None


def test_bf16_parameters_keep_fp32_weight_gradients():
    x = torch.randn(8, 4, dtype=torch.bfloat16, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(4, 6, dtype=torch.bfloat16))
    if not hasattr(w, 'grad_dtype'):
        pytest.skip('requires parameter grad_dtype support')
    w.grad_dtype = torch.float32
    ref_w = torch.nn.Parameter(w.detach().clone())
    ref_w.grad_dtype = torch.float32
    out, region = run(x, w, [])
    expected = Linear.apply(x * 2, ref_w, None, [])
    out.sum().backward()
    expected.sum().backward()
    assert w.grad.dtype == torch.float32
    torch.testing.assert_close(w.grad, ref_w.grad, rtol=0, atol=0)
    assert not region.tasks


def test_callback_failure_clears_the_current_graph_task():
    x = torch.randn(8, 4, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(4, 6))
    out, region = run(x, w, [])
    original = region.defer

    def defer(indices, callback, inputs):
        def fail():
            raise ValueError('weight failure')
        original(indices, fail, inputs)

    region.defer = defer
    with pytest.raises(ValueError, match='weight failure'):
        out.sum().backward()
    assert not region.tasks


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_weight_gemm_waits_for_inputs_but_overlaps_pending_communication():
    from nnscaler.runtime.stream import _get_module_stream
    device = torch.device('cuda', torch.cuda.current_device())
    x = torch.randn(64, 64, device=device, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(64, 64, device=device))
    rx, rw = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    side = _get_module_stream(device, 'weights_cuda_test')
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm = x.detach().t() @ x.detach()
    torch.cuda.synchronize()
    del warm
    inputs_ready, comm_done = torch.cuda.Event(), torch.cuda.Event()
    comm = torch.cuda.Stream()

    class Communication(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.point = capture_async_backward()
            return value * 2

        @staticmethod
        def backward(ctx, dy):
            torch.cuda._sleep(500_000_000)
            inputs_ready.record()
            ctx.point.record_ready()
            comm.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(comm):
                torch.cuda._sleep(1_000_000_000)
                comm_done.record()
            try:
                ctx.point.submitted()
            finally:
                torch.cuda.current_stream().wait_event(comm_done)
            return dy * 2

    region = WeightBackwardRegion('weights_cuda_test', (w,))
    original = region.defer

    def defer(indices, callback, inputs):
        def check():
            gradients = callback()
            finished = torch.cuda.Event()
            finished.record()
            finished.synchronize()
            assert inputs_ready.query(), 'dW ran before its input readiness'
            assert not comm_done.query(), 'dW waited for communication completion'
            return gradients
        original(indices, check, inputs)

    region.defer = defer
    with region.anchor():
        routed = Communication.apply(prepare_async_input(x))
    with region.compute():
        binding = prepare_weight_backward((w,))
        out = Linear.apply(routed, binding.parameters[0], binding, [])
    region.close_forward()
    out.sum().backward()
    ((rx * 2) @ rw).sum().backward()
    torch.testing.assert_close(x.grad, rx.grad)
    torch.testing.assert_close(w.grad, rw.grad)
    assert not region.tasks


def test_parameter_updates_with_old_outputs_alive_use_fresh_weights():
    x = torch.randn(8, 4, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(4, 6))
    rx = x.detach().clone().requires_grad_()
    rw = torch.nn.Parameter(w.detach().clone())
    old_outputs = []
    for _ in range(3):
        x.grad = w.grad = rx.grad = rw.grad = None
        out, region = run(x, w, [])
        ref = (rx * 2) @ rw
        out.square().sum().backward()
        ref.square().sum().backward()
        torch.testing.assert_close(out, ref)
        torch.testing.assert_close(w.grad, rw.grad)
        torch.testing.assert_close(x.grad, rx.grad)
        with torch.no_grad():
            w.add_(w.grad, alpha=-.003)
            rw.add_(rw.grad, alpha=-.003)
        old_outputs.append(out)
        assert not region.tasks


def test_callback_restores_registration_autocast_context():
    x = torch.randn(8, 4, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(4, 6))
    out, region = run(x, w, [])
    original = region.defer

    def defer(indices, callback, inputs):
        def check():
            assert torch.is_autocast_enabled('cpu')
            return callback()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            original(indices, check, inputs)

    region.defer = defer
    out.sum().backward()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected = (x.detach() * 2).t() @ torch.ones_like(out)
    torch.testing.assert_close(w.grad, expected.float(), rtol=0, atol=0)


def test_higher_order_weight_derivatives_fail_explicitly():
    x = torch.randn(8, 4, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(4, 6))
    out, region = run(x, w, [])
    with pytest.raises(RuntimeError, match='first-order'):
        torch.autograd.grad(out.sum(), w, create_graph=True)
    assert not region.tasks


def test_real_reducer_counts_and_mixed_native_contributions():
    from unittest.mock import Mock
    from nnscaler.runtime.adapter.reducer import Reducer
    from ..utils import mock_reducer_env
    with mock_reducer_env(0, 2):
        w = torch.nn.Parameter(torch.randn(4, 6))
        rw = torch.nn.Parameter(w.detach().clone())
        reducer = Reducer([0, 1], async_op=True)
        reducer.add_param(w)
        reducer.build_buckets()
        reducer.set_async_grad_expected_counts({w: 2})
        bucket = reducer.buckets[0]
        bucket._launch_async_reduce = Mock()
        for step in range(2):
            x = torch.randn(8, 4, requires_grad=True)
            rx = x.detach().clone().requires_grad_()
            output, region = run(x, w, [])
            (output.square().sum() + w.square().sum() * .3).backward()
            (((rx * 2) @ rw).square().sum() + rw.square().sum() * .3).backward()
            offset = bucket._pofset[w]
            actual = bucket._contiguous_grads[offset:offset + w.numel()].view_as(w)
            torch.testing.assert_close(actual, rw.grad)
            torch.testing.assert_close(x.grad, rx.grad)
            assert bucket._async_seen_param_cnt[w] == step + 1
            assert w.grad is None and not region.tasks
            assert bucket._launch_async_reduce.call_count == int(step == 1)
