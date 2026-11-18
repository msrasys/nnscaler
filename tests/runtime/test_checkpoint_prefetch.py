import gc

import pytest
import torch
from torch.utils.checkpoint import checkpoint, CheckpointError

import nnscaler
from nnscaler.runtime.checkpoint_prefetch import Budget, Prefetch


@pytest.mark.parametrize('when', ['forward', 'backward'])
def test_prefetched_values_preserve_math_and_repeat_backward(when):
    torch.manual_seed(19)
    x = torch.randn(8, 8, requires_grad=True)
    weight = torch.randn(8, 8, requires_grad=True)
    rx, rw = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()
    calls = []
    budget = Budget(1 << 20)
    point = Prefetch('test', budget)

    def function(value):
        calls.append(True)
        return ((value @ weight).sin() @ weight.t()).cos()

    class Anchor(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.point = nnscaler.capture_async_backward()
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            ctx.point.record_ready()
            ctx.point.submitted()
            return grad

    y = checkpoint(function, x, use_reentrant=False,
                   context_fn=lambda: point.contexts((x, weight)))
    expected = ((rx @ rw).sin() @ rw.t()).cos()
    if when == 'forward':
        assert point.prefetch()
        point.finish_forward()
        assert len(calls) == 2
        output = y
    else:
        with point:
            output = Anchor.apply(y)
    for iteration in range(2):
        output.sum().backward(retain_graph=iteration == 0)
        expected.sum().backward(retain_graph=iteration == 0)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, rw.grad, rtol=0, atol=0)
        assert budget.used == 0
        assert len(calls) == iteration + 2
        x.grad = weight.grad = rx.grad = rw.grad = None


def test_budget_falls_back_to_normal_recompute():
    point = Prefetch('test', Budget(0))
    x = torch.randn(4, requires_grad=True)
    calls = []
    def f(x):
        calls.append(True)
        return x.sin().square()
    y = checkpoint(f, x, use_reentrant=False, context_fn=lambda: point.contexts((x,)))
    assert not point.prefetch()
    assert len(calls) == 1
    y.sum().backward()
    assert len(calls) == 2
    assert point.budget.used == 0


def test_prefetch_restores_rng_and_releases_abandoned_graph():
    torch.manual_seed(71)
    x = torch.randn(16, requires_grad=True)
    budget = Budget(1 << 20)
    point = Prefetch('test', budget)
    y = checkpoint(lambda v: (v * torch.rand_like(v)).sin(), x, use_reentrant=False,
                   context_fn=lambda: point.contexts((x,)))
    rng = torch.get_rng_state()
    assert point.prefetch()
    assert torch.equal(torch.get_rng_state(), rng)
    assert budget.used
    del y, point
    gc.collect()
    assert budget.used == 0


def test_metadata_failure_releases_reservation():
    point = Prefetch('test', Budget(1 << 20))
    x = torch.randn(4, requires_grad=True)
    calls = []
    def f(x):
        calls.append(True)
        return x.sin() if len(calls) == 1 else x[:2].sin()
    y = checkpoint(f, x, use_reentrant=False, context_fn=lambda: point.contexts((x,)))
    with pytest.raises(CheckpointError):
        point.prefetch()
    assert point.budget.used == 0


def test_input_version_failure_releases_reservation():
    point = Prefetch('test', Budget(1 << 20))
    x = torch.randn(4, requires_grad=True)
    y = checkpoint(lambda v: v.sin().square(), x, use_reentrant=False,
                   context_fn=lambda: point.contexts((x,)))
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match='inplace'):
        point.prefetch()
    assert point.budget.used == 0


def test_changed_storage_capacity_cannot_bypass_budget():
    point = Prefetch('test', Budget(1024))
    x = torch.randn(4, requires_grad=True)
    calls = []
    def function(value):
        calls.append(True)
        hidden = value.sin()
        if len(calls) > 1:
            hidden = torch.cat((hidden, torch.zeros(256)))[:4]
        return hidden.square()
    y = checkpoint(function, x, use_reentrant=False,
                   context_fn=lambda: point.contexts((x,)))
    with pytest.raises(RuntimeError, match='storage footprint'):
        point.prefetch()
    assert point.budget.used == 0


def test_inference_has_no_prefetch_work():
    point = Prefetch('test', Budget(1024))
    x = torch.randn(4)
    with torch.no_grad():
        y = checkpoint(lambda v: v.sin().square(), x, use_reentrant=False,
                       context_fn=lambda: point.contexts((x,)))
    assert not point.prefetch()
    torch.testing.assert_close(y, x.sin().square())
    assert point.budget.used == 0


def test_prefetch_point_cannot_alias_two_live_forwards():
    point = Prefetch('test', Budget(1024))
    x = torch.randn(4, requires_grad=True)
    y = checkpoint(lambda v: v.sin().square(), x, use_reentrant=False,
                   context_fn=lambda: point.contexts((x,)))
    with pytest.raises(RuntimeError, match='one forward invocation'):
        checkpoint(lambda v: v.sin().square(), x, use_reentrant=False,
                   context_fn=lambda: point.contexts((x,)))


def test_higher_order_rejected_without_leaking_prefetch_budget():
    point = Prefetch('test', Budget(1024))
    x = torch.randn(4, requires_grad=True)
    y = checkpoint(lambda v: v.sin().square(), x, use_reentrant=False,
                   context_fn=lambda: point.contexts((x,)))
    assert point.prefetch()
    with pytest.raises(RuntimeError, match='first-order'):
        torch.autograd.grad(y.sum(), x, create_graph=True)
    assert point.budget.used == 0


def test_nested_checkpoint_prefetch_rejected():
    point = Prefetch('test', Budget(1024))
    x = torch.randn(4, requires_grad=True)
    with pytest.raises(RuntimeError, match='nested'):
        checkpoint(lambda value: checkpoint(lambda v: v.sin().square(), value,
                   use_reentrant=False, context_fn=lambda: point.contexts((value,))),
                   x, use_reentrant=False)


def test_partial_backward_releases_cache_and_retained_graph_recomputes():
    x = torch.randn(8, requires_grad=True)
    point = Prefetch('partial', Budget(1 << 20))
    calls = []

    def function(value):
        calls.append(True)
        return value.sin().square(), value.cos().square()

    first, second = checkpoint(function, x, use_reentrant=False,
                               context_fn=lambda: point.contexts((x,)))
    assert point.prefetch()
    dx, = torch.autograd.grad(first.sum(), x, retain_graph=True)
    torch.testing.assert_close(dx, 2 * x.sin() * x.cos())
    assert point.budget.used == 0 and len(calls) == 2
    dx, = torch.autograd.grad(second.sum(), x)
    torch.testing.assert_close(dx, -2 * x.sin() * x.cos())
    assert point.budget.used == 0 and len(calls) == 3


def test_discarded_forward_does_not_depend_on_cyclic_gc_for_budget_release():
    x = torch.randn(16, requires_grad=True)
    budget = Budget(1 << 20)
    point = Prefetch('discarded', budget)
    y = checkpoint(lambda v: v.sin().square(), x, use_reentrant=False,
                   context_fn=lambda: point.contexts((x,)))
    assert point.prefetch() and budget.used
    del y, point
    assert budget.used == 0


def test_missing_backend_capture_is_rejected():
    point = Prefetch('missing_capture', Budget(1024))
    with pytest.raises(RuntimeError, match='did not capture'):
        with point:
            pass


def test_missing_backward_readiness_is_rejected_before_replay():
    point = Prefetch('missing_ready', Budget(1024))
    x = torch.randn(4, requires_grad=True)
    y = checkpoint(lambda v: v.sin().square(), x, use_reentrant=False,
                   context_fn=lambda: point.contexts((x,)))

    class BrokenAnchor(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.point = nnscaler.capture_async_backward()
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            ctx.point.submitted()
            return grad

    with point:
        result = BrokenAnchor.apply(y)
    with pytest.raises(RuntimeError, match='input readiness'):
        result.sum().backward()
    assert point.budget.used == 0 and point.stats['launched'] == 0
