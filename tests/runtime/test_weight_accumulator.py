# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import weakref

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nnscaler import WeightGradientAccumulator, prepare_weight_backward


class FragmentLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, binding, records, kind, delay):
        ctx.save_for_backward(x, weight)
        ctx.binding, ctx.records, ctx.kind, ctx.delay = binding, records, kind, delay
        return x @ weight

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dx = dy @ weight.t()
        if ctx.binding is None:
            return dx, x.float().t() @ dy.float(), None, None, None, None

        def direct(outputs):
            if outputs is None:
                outputs = (torch.zeros_like(weight, dtype=torch.float32),)
            if ctx.delay:
                torch.cuda._sleep(ctx.delay)
            ctx.records.append(outputs[0].data_ptr())
            outputs[0].addmm_(x.float().t(), dy.float())
            if ctx.kind == 'failure':
                raise ValueError('backend failure')
            if ctx.kind == 'replace':
                return (outputs[0].clone(),)
            if ctx.kind == 'shape':
                return (outputs[0].resize_(1),)
            return outputs

        if ctx.kind == 'legacy':
            ctx.binding.defer(lambda: (x.float().t() @ dy.float(),), (x, dy))
        else:
            ctx.binding.accumulate(direct, (x, dy))
        return dx, None, None, None, None, None


def fragment(x, weight, records, kind='direct', delay=0):
    binding = prepare_weight_backward((weight,))
    return FragmentLinear.apply(x, weight if binding is None else binding.parameters[0],
                                binding, records, kind, delay)


def run(xs, weight, records, *, recompute=False, kind='direct', delay=0):
    region = WeightGradientAccumulator('accumulator_test', (weight,))
    def compute(x):
        with region.compute():
            return fragment(x, weight, records, kind, delay)
    with region:
        outputs = [checkpoint(compute, x, use_reentrant=False) if recompute else compute(x) for x in xs]
    return outputs, region


@pytest.mark.parametrize('kind', ['direct', 'legacy'])
@pytest.mark.parametrize('recompute', [False, True])
def test_fragment_gradients_publish_once_and_support_repeated_backward(kind, recompute):
    torch.manual_seed(7)
    weight = torch.nn.Parameter(torch.randn(4, 6))
    ref_weight = weight.detach().clone().requires_grad_()
    xs = [torch.randn(n, 4, requires_grad=True) for n in (3, 5, 2)]
    refs = [x.detach().clone().requires_grad_() for x in xs]
    records, hooks = [], []
    weight.register_post_accumulate_grad_hook(lambda p: hooks.append(p.grad.clone()))
    actual, region = run(xs, weight, records, recompute=recompute, kind=kind)
    expected = [x @ ref_weight for x in refs]
    for iteration in range(2):
        sum(y.square().sum() for y in actual).backward(retain_graph=iteration == 0)
        sum(y.square().sum() for y in expected).backward(retain_graph=iteration == 0)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(weight.grad, ref_weight.grad)
        for x, ref in zip(xs, refs):
            torch.testing.assert_close(x.grad, ref.grad)
        assert not region.tasks
        if kind == 'direct':
            assert len(set(records[3*iteration:3*(iteration+1)])) == 1
    assert len(hooks) == 2


def test_pruned_parameter_edges_do_not_accumulate_and_release_the_region():
    x = torch.randn(4, 4, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(4, 6))
    actual, region = run([x], weight, [])
    dx, = torch.autograd.grad(actual[0].sum(), (x,))
    torch.testing.assert_close(dx, torch.ones_like(actual[0]) @ weight.t())
    assert weight.grad is None and not region.tasks
    reference = weakref.ref(region)
    del actual, region
    assert reference() is None


def test_live_forwards_have_separate_buffers():
    weight = torch.nn.Parameter(torch.randn(4, 6))
    ref_weight = weight.detach().clone().requires_grad_()
    inputs = [torch.randn(5, 4, requires_grad=True) for _ in range(2)]
    records = [[], []]
    calls = [run([x], weight, record) for x, record in zip(inputs, records)]
    sum(outputs[0].sum() for outputs, _ in calls).backward()
    sum((x.detach() @ ref_weight).sum() for x in inputs).backward()
    torch.testing.assert_close(weight.grad, ref_weight.grad)
    assert records[0][0] != records[1][0]
    assert all(not region.tasks for _, region in calls)


@pytest.mark.parametrize('kind,match', [('failure', 'backend failure'),
                                       ('replace', 'preserve'), ('shape', 'preserve')])
def test_failed_callbacks_do_not_publish_partial_buffers(kind, match):
    x = torch.randn(4, 4, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(4, 6))
    actual, region = run([x], weight, [], kind=kind)
    with pytest.raises((ValueError, RuntimeError), match=match):
        actual[0].sum().backward()
    assert weight.grad is None and not region.tasks and region.failed


def test_forward_scope_rejects_reuse_and_unbound_parameters():
    x = torch.randn(4, 4, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(4, 6))
    _, region = run([x], weight, [])
    with pytest.raises(RuntimeError, match='one forward'):
        with region:
            pass
    with pytest.raises(RuntimeError, match='must be bound'):
        with WeightGradientAccumulator('unused', (weight,)):
            pass
    fresh = WeightGradientAccumulator('no_anchor', (weight,))
    with pytest.raises(RuntimeError, match='does not use a communication anchor'):
        fresh.anchor()
    assert not fresh.aliases


def test_higher_order_backward_fails_explicitly():
    x = torch.randn(4, 4, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(4, 6))
    actual, region = run([x], weight, [])
    with pytest.raises(RuntimeError, match='first-order'):
        torch.autograd.grad(actual[0].sum(), (x, weight), create_graph=True)
    assert not region.tasks


def test_accumulating_backend_keeps_ordinary_deferred_region_protocol():
    from nnscaler import prepare_async_input
    from nnscaler.runtime.weight_backward import WeightBackwardRegion
    from .test_weight_backward import Anchor
    x = torch.randn(4, 4, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(4, 6))
    order, records = [], []
    region = WeightBackwardRegion('ordinary_weights', (weight,))
    with region.anchor():
        value = Anchor.apply(prepare_async_input(x), order)
    with region.compute():
        y = fragment(value, weight, records)
    region.close_forward()
    y.sum().backward()
    torch.testing.assert_close(weight.grad, (2*x.detach()).t() @ torch.ones_like(y))
    assert order == ['submit'] and len(records) == 1 and not region.tasks


def test_reducer_counts_one_contribution_for_multiple_fragments():
    from unittest.mock import Mock
    from nnscaler.runtime.adapter.reducer import Reducer
    from ..utils import mock_reducer_env
    with mock_reducer_env(0, 2):
        weight = torch.nn.Parameter(torch.randn(4, 6))
        ref_weight = torch.nn.Parameter(weight.detach().clone())
        reducer = Reducer([0, 1], async_op=True)
        reducer.add_param(weight)
        reducer.build_buckets()
        reducer.set_async_grad_expected_counts({weight: 2})
        bucket = reducer.buckets[0]
        bucket._launch_async_reduce = Mock()
        for step in range(2):
            x = torch.randn(8, 4, requires_grad=True)
            rx = x.detach().clone().requires_grad_()
            outputs, region = run([x[:3], x[3:]], weight, [])
            (sum(y.square().sum() for y in outputs) + weight.square().sum() * .3).backward()
            ((rx @ ref_weight).square().sum() + ref_weight.square().sum() * .3).backward()
            offset = bucket._pofset[weight]
            actual = bucket._contiguous_grads[offset:offset+weight.numel()].view_as(weight)
            torch.testing.assert_close(actual, ref_weight.grad)
            torch.testing.assert_close(x.grad, rx.grad)
            assert bucket._async_seen_param_cnt[weight] == step + 1
            assert bucket._launch_async_reduce.call_count == int(step == 1)
            assert weight.grad is None and not region.tasks


def test_disabled_grad_mode_uses_the_backend_ordinary_path():
    weight = torch.nn.Parameter(torch.randn(4, 6))
    x = torch.randn(3, 4, requires_grad=True)
    records = []
    with torch.no_grad():
        outputs, region = run([x], weight, records)
        torch.testing.assert_close(outputs[0], x @ weight)
    assert not outputs[0].requires_grad and not region.enabled and not region.tasks and not records


def test_frozen_backend_parameter_has_no_accumulation_buffer():
    class Mixed(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, live, frozen, binding):
            ctx.save_for_backward(x, live, frozen)
            ctx.binding = binding
            return x @ (live + frozen)

        @staticmethod
        def backward(ctx, dy):
            x, live, frozen = ctx.saved_tensors
            def weight(outputs):
                assert outputs[1] is None
                outputs[0].addmm_(x.t(), dy)
                return outputs
            ctx.binding.accumulate(weight, (x, dy))
            return dy @ (live + frozen).t(), None, None, None

    x = torch.randn(4, 3, requires_grad=True)
    live = torch.nn.Parameter(torch.randn(3, 5))
    frozen = torch.nn.Parameter(torch.randn(3, 5), requires_grad=False)
    with WeightGradientAccumulator('mixed', (live,)) as region:
        binding = prepare_weight_backward((live, frozen))
        y = Mixed.apply(x, *binding.parameters, binding)
    y.sum().backward()
    torch.testing.assert_close(live.grad, x.detach().t() @ torch.ones_like(y))
    assert frozen.grad is None and not region.tasks


@pytest.mark.parametrize('name', ['', None, 1, 'default'])
def test_accumulator_requires_a_named_stream(name):
    with pytest.raises(ValueError, match='named stream'):
        WeightGradientAccumulator(name, (torch.nn.Parameter(torch.ones(2)),))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cuda_accumulation_preserves_fp32_parameter_gradients(dtype):
    weight = torch.nn.Parameter(torch.randn(32, 32, device='cuda', dtype=dtype))
    weight.grad_dtype = torch.float32
    xs = [torch.randn(n, 32, device='cuda', dtype=dtype, requires_grad=True) for n in (64, 32)]
    records = []
    ys, region = run(xs, weight, records, recompute=True)
    sum(y.float().sum() for y in ys).backward()
    expected = sum(x.detach().float().t() @ torch.ones_like(y, dtype=torch.float32)
                   for x, y in zip(xs, ys))
    torch.testing.assert_close(weight.grad, expected)
    assert weight.grad.dtype == torch.float32 and len(set(records)) == 1 and not region.tasks


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_input_backward_can_advance_before_weight_work_finishes():
    x = torch.randn(32, 32, device='cuda', requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(32, 32, device='cuda'))
    warm, _ = run([x], weight, [])
    warm[0].sum().backward()
    torch.cuda.synchronize()
    del warm
    x.grad = weight.grad = None
    dx_ready = torch.cuda.Event(enable_timing=True)
    dw_ready = torch.cuda.Event(enable_timing=True)
    outputs, region = run([x], weight, [], delay=20_000_000)
    x.register_hook(lambda grad: dx_ready.record())
    weight.register_post_accumulate_grad_hook(lambda param: dw_ready.record())
    outputs[0].sum().backward()
    torch.cuda.synchronize()
    assert dx_ready.elapsed_time(dw_ready) > 1., 'dInput was blocked behind dWeight'
    assert not region.tasks


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_contributors_on_distinct_producer_streams():
    caller = torch.cuda.current_stream()
    producers = [torch.cuda.Stream(), torch.cuda.Stream()]
    weight = torch.nn.Parameter(torch.randn(32, 32, device='cuda'))
    originals = [torch.randn(n, 32, device='cuda', requires_grad=True) for n in (64, 48)]
    outputs, records = [], []
    with WeightGradientAccumulator('multi_producer_weights', (weight,)) as region:
        for value, producer in zip(originals, producers):
            producer.wait_stream(caller)
            with torch.cuda.stream(producer):
                value.record_stream(producer)
                weight.record_stream(producer)
                # The temporary is owned by a producer stream but read by dW.
                outputs.append(fragment(value.sin(), weight, records, delay=1_000_000))
            caller.wait_stream(producer)
            outputs[-1].record_stream(caller)
    sum(output.sum() for output in outputs).backward()
    expected = sum(value.detach().sin().t() @ torch.ones_like(output)
                   for value, output in zip(originals, outputs))
    torch.testing.assert_close(weight.grad, expected)
    assert len(set(records)) == 1 and not region.tasks
