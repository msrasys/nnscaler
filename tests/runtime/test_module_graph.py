# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import gc
import weakref

import pytest
import torch

from nnscaler.runtime.module_graph import ModuleGraph, run_module_graph, prepare_module_graphs
from nnscaler.runtime.stream import drain_stream_holds

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def transform(x, weight):
    return (x @ weight).sin()


@pytest.mark.parametrize('capture_backward', [True, False])
@pytest.mark.parametrize('order', [(0, 1, 2), (2, 0, 1)])
def test_live_forwards_arbitrary_backward_order_and_retained_values(order, capture_backward):
    torch.manual_seed(31)
    weight = torch.nn.Parameter(torch.randn(16, 16, device='cuda'))
    ref_weight = torch.nn.Parameter(weight.detach().clone())
    sample = torch.zeros(8, 16, device='cuda', requires_grad=True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        pool = ModuleGraph(transform, (sample, weight), (1,), capacity=3, stream=stream, capture_backward=capture_backward)
        xs = [torch.randn_like(sample, requires_grad=True) for _ in range(3)]
        refs = [x.detach().clone().requires_grad_() for x in xs]
        actual = [pool(x, weight) for x in xs]
        expected = [transform(x, ref_weight) for x in refs]
        snapshots = [v.detach().clone() for v in actual]
        for i in order:
            actual[i].square().sum().backward()
            expected[i].square().sum().backward()
            torch.testing.assert_close(xs[i].grad, refs[i].grad)
        torch.testing.assert_close(weight.grad, ref_weight.grad)
        torch.testing.assert_close(actual, expected)
        for _ in range(4):
            value = pool(torch.randn_like(sample, requires_grad=True), weight)
            value.sum().backward()
        torch.testing.assert_close(actual, snapshots)
        assert len(pool.free) == 3
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)


@pytest.mark.parametrize('capture_backward', [True, False])
def test_parameter_hooks_apply_once_and_weight_updates_are_visible(capture_backward):
    weight = torch.nn.Parameter(torch.randn(16, 16, device='cuda'))
    hook_calls = []
    weight.register_hook(lambda grad: hook_calls.append(grad.clone()) or grad * .5)
    sample = torch.randn(8, 16, device='cuda', requires_grad=True)
    pool = ModuleGraph(transform, (sample, weight), (1,), capture_backward=capture_backward)
    assert not hook_calls
    for _ in range(2):
        with torch.no_grad():weight.add_(.01)
        x = sample.detach().clone().requires_grad_()
        ref_x = x.detach().clone().requires_grad_()
        ref_w = weight.detach().clone().requires_grad_()
        actual = pool(x, weight)
        expected = transform(ref_x, ref_w)
        observed = torch.autograd.grad(actual.square().sum(), (x, weight))
        reference = torch.autograd.grad(expected.square().sum(), (ref_x, ref_w))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(observed[0], reference[0])
        torch.testing.assert_close(observed[1], reference[1] * .5)
    assert len(hook_calls) == 2
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)


@pytest.mark.parametrize('capture_backward', [True, False])
def test_aliased_inputs_have_independent_gradient_edges(capture_backward):
    def function(a, b, weight):
        return (a @ weight).sin() + b.cos()

    x = torch.randn(8, 16, device='cuda', requires_grad=True)
    w = torch.nn.Parameter(torch.randn(16, 16, device='cuda'))
    pool = ModuleGraph(function, (x, x, w), (2,), capture_backward=capture_backward)
    actual = pool(x, x, w)
    observed = torch.autograd.grad(actual.sum(), (x, w))
    expected = function(x, x, w)
    reference = torch.autograd.grad(expected.sum(), (x, w))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(observed, reference)
    with pytest.raises(ValueError, match='aliases'):
        pool(x, x.clone(), w)
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)


@pytest.mark.parametrize('capture_backward', [True, False])
def test_calibration_capacity_and_abandoned_forward_release(capture_backward):
    owner = torch.nn.Module()
    w = torch.nn.Parameter(torch.randn(16, 16, device='cuda'))
    xs = [torch.randn(8, 16, device='cuda', requires_grad=True) for _ in range(3)]
    calls = [run_module_graph(owner, 'test', transform, (x, w), (1,), capture_backward=capture_backward) for x in xs]
    for y in calls:y.sum().backward()
    prepare_module_graphs(owner)
    enabled = gc.isenabled()
    try:
        gc.disable()
        calls = [run_module_graph(owner, 'test', transform, (x, w), (1,), capture_backward=capture_backward) for x in xs]
        with pytest.raises(RuntimeError, match='capacity'):
            run_module_graph(owner, 'test', transform, (xs[0], w), (1,), capture_backward=capture_backward)
        del calls
        y = run_module_graph(owner, 'test', transform, (xs[0], w), (1,), capture_backward=capture_backward)
        y.sum().backward()
    finally:
        if enabled:gc.enable()
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)


@pytest.mark.parametrize('capture_backward', [True, False])
def test_interleaving_reuse_keeps_other_slots_custom_backward_state(capture_backward):
    class HiddenState(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.factor = 3*x+1  # Deliberately stored outside save_for_backward.
            return 1.5*x.square()+x

        @staticmethod
        def backward(ctx, grad):
            return grad*ctx.factor

    def function(x, w):return HiddenState.apply(x@w)

    w = torch.nn.Parameter(torch.randn(16,16,device='cuda'))
    sample = torch.zeros(8,16,device='cuda',requires_grad=True)
    pool = ModuleGraph(function,(sample,w),(1,),capacity=2, capture_backward=capture_backward)
    pending=[]
    for i in range(8):
        x = torch.randn_like(sample,requires_grad=True)
        pending.append((x,pool(x,w)))
        if len(pending)==2 or i==7:
            old,y=pending.pop(0)
            expected=function(old,w)
            actual=torch.autograd.grad(y.sum(),(old,w))
            reference=torch.autograd.grad(expected.sum(),(old,w))
            torch.testing.assert_close(y,expected)
            torch.testing.assert_close(actual,reference)
    for x,y in pending:y.sum().backward()
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)
