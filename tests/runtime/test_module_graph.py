# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import gc
import weakref

import pytest
import torch

from nnscaler.runtime.module_graph import ModuleGraph, run_module_graph, prepare_module_graph, prepare_module_graphs
from nnscaler.runtime.stream import drain_stream_holds

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def transform(x, weight):
    return (x @ weight).sin()


def test_single_use_backward_state_replays_across_slots():
    class SingleUse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.used = False
            ctx.save_for_backward(x)
            return x.sin()

        @staticmethod
        def backward(ctx, gradient):
            if ctx.used:
                raise RuntimeError('Backward state may be consumed only once')
            ctx.used = True
            return gradient * ctx.saved_tensors[0].cos()

    def function(x, weight):
        return SingleUse.apply(x + weight)

    weight = torch.nn.Parameter(torch.randn(256, 64, device='cuda'))
    sample = torch.randn_like(weight, requires_grad=True)
    probe = function(sample, weight)
    torch.autograd.grad(probe.sum(), (sample, weight), retain_graph=True)
    with pytest.raises(RuntimeError, match='only once'):
        torch.autograd.grad(probe.sum(), (sample, weight), retain_graph=True)
    del probe
    pool = ModuleGraph(function, (sample, weight), (1,), capacity=2)
    for _ in range(3):
        with torch.no_grad():
            weight.add_(.01)
        inputs = [torch.randn_like(sample, requires_grad=True) for _ in range(2)]
        actual = [pool(x, weight) for x in inputs]
        expected = [function(x, weight) for x in inputs]
        for i in (1, 0):
            observed = torch.autograd.grad(actual[i].sum(), (inputs[i], weight))
            reference = torch.autograd.grad(expected[i].sum(), (inputs[i], weight))
            torch.testing.assert_close(actual[i], expected[i])
            torch.testing.assert_close(observed, reference)
        assert len(pool.free) == 2
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)


@pytest.mark.parametrize('capture_backward', [True, False])
def test_preparation_does_not_replay_or_create_autograd_edges(monkeypatch, capture_backward):
    from nnscaler.runtime.module_graph import _profiles
    owner = torch.nn.Module()
    weight = torch.nn.Parameter(torch.randn(16, 16, device='cuda'))
    xs = [torch.randn(8, 16, device='cuda', requires_grad=True) for _ in range(3)]
    calls = [run_module_graph(owner, 'test', transform, (x, weight), (1,), capture_backward=capture_backward)
             for x in xs]
    for y in calls:
        y.sum().backward()
    prepare_module_graphs(owner)
    replays = []
    replay = torch.cuda.CUDAGraph.replay
    monkeypatch.setattr(torch.cuda.CUDAGraph, 'replay', lambda graph: (replays.append(graph), replay(graph))[1])
    prepared = [prepare_module_graph(owner, 'test', (x, weight)) for x in xs]
    assert not replays
    assert all(not value.requires_grad for call in prepared for value in call.ticket.outputs)
    actual = [run_module_graph(owner, 'test', transform, (x, weight), (1,),
                               capture_backward=capture_backward, prepared=call)
              for x, call in zip(xs, prepared)]
    assert len(replays) == 3
    for i in (2, 0, 1):
        reference = transform(xs[i], weight)
        torch.testing.assert_close(actual[i], reference)
        observed = torch.autograd.grad(actual[i].sum(), (xs[i], weight))
        expected = torch.autograd.grad(reference.sum(), (xs[i], weight))
        torch.testing.assert_close(observed, expected)
    assert len(_profiles[owner]['test'].pool.free) == 3
    with pytest.raises(ValueError, match='different module graph'):
        run_module_graph(owner, 'test', transform, (xs[0], weight), (1,),
                         capture_backward=capture_backward, prepared=prepared[0])


def test_abandoned_and_mutated_preparations_release_the_slot():
    from nnscaler.runtime.module_graph import _profiles
    owner = torch.nn.Module()
    weight = torch.nn.Parameter(torch.randn(16, 16, device='cuda'))
    x = torch.randn(8, 16, device='cuda', requires_grad=True)
    run_module_graph(owner, 'test', transform, (x, weight), (1,), capture_backward=False).sum().backward()
    prepare_module_graphs(owner)
    pool = _profiles[owner]['test'].pool
    enabled = gc.isenabled()
    gc.disable()
    try:
        call = prepare_module_graph(owner, 'test', (x, weight))
        assert not pool.free
        del call
        assert len(pool.free) == 1
        call = prepare_module_graph(owner, 'test', (x, weight))
        with torch.no_grad():
            x.add_(1)
        with pytest.raises(RuntimeError, match='changed before execution'):
            run_module_graph(owner, 'test', transform, (x, weight), (1,),
                             capture_backward=False, prepared=call)
        assert len(pool.free) == 1
        for _ in range(4):
            call = prepare_module_graph(owner, 'test', (x, weight))
            y = run_module_graph(owner, 'test', transform, (x, weight), (1,),
                                 capture_backward=False, prepared=call)
            ref = weakref.ref(y)
            y.sum().backward()
            del y
            assert ref() is None
            assert len(pool.free) == 1
    finally:
        if enabled:
            gc.enable()
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)


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
        torch.testing.assert_close([x.grad for x in xs], [x.grad for x in refs])
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
