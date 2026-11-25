# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import weakref

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nnscaler import WeightGradientAccumulator, prepare_weight_backward


class CommunicationAnchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, order):
        from nnscaler.runtime.backward import capture_async_backward
        ctx.point = capture_async_backward()
        ctx.order = order
        return x * 2

    @staticmethod
    def backward(ctx, dy):
        if ctx.point is not None:
            ctx.point.record_ready()
        ctx.order.append('communication')
        return dy * 2, None


@pytest.mark.parametrize('recompute', [False, True])
@pytest.mark.parametrize('only_weight', [False, True])
def test_deferred_weights_follow_communication_without_changing_gradients(recompute, only_weight):
    torch.manual_seed(173)
    w = torch.nn.Parameter(torch.randn(4, 6))
    ref_w = w.detach().clone().requires_grad_()
    x = torch.randn(5, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    order = []
    region = WeightGradientAccumulator('deferred', (w,), defer_until_backward=True)
    controlled = region.after_backward(x)
    with region.backward_point():
        h = CommunicationAnchor.apply(controlled, order)

    def compute(h):
        with region.compute():
            return fragment(h, w, order)

    y = checkpoint(compute, h, use_reentrant=False) if recompute else compute(h)
    region.close_forward()
    expected = (ref_x * 2) @ ref_w
    if only_weight:
        actual_grads = torch.autograd.grad(y.square().sum(), (w,))
        expected_grads = torch.autograd.grad(expected.square().sum(), (ref_w,))
        torch.testing.assert_close(actual_grads, expected_grads)
        assert x.grad is w.grad is None
    else:
        y.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(w.grad, ref_w.grad)
    assert len(order) == 2 and order[0] == 'communication' and isinstance(order[1], int)
    assert not region.tasks


def test_deferred_pruned_weights_release_inputs_without_running_callbacks():
    x = torch.randn(5, 4, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(4, 6))
    order = []
    region = WeightGradientAccumulator('pruned_deferred', (w,), defer_until_backward=True)
    h = CommunicationAnchor.apply(region.after_backward(x), order)
    with region.operator():
        y = fragment(h, w, order)
    dx, = torch.autograd.grad(y.sum(), (x,))
    torch.testing.assert_close(dx, torch.ones_like(y) @ w.t() * 2)
    assert order == ['communication'] and w.grad is None and not region.tasks


def test_early_anchor_failure_releases_deferred_aliases_with_gc_disabled():
    import gc
    from nnscaler.runtime.weight_accumulator import weight_backward_scopes
    references = []
    enabled = gc.isenabled()

    def fail():
        with weight_backward_scopes() as regions:
            w = torch.nn.Parameter(torch.randn(4, 6))
            region = WeightGradientAccumulator('failed_deferred', (w,), defer_until_backward=True)
            regions.append(region)
            references.append(weakref.ref(region))
            region.after_backward(torch.randn(5, 4, requires_grad=True))
            raise ValueError('anchor failed before compute')

    try:
        gc.disable()
        try:
            fail()
        except ValueError:
            pass
        assert references[0]() is None
    finally:
        if enabled:gc.enable()


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
    return FragmentLinear.apply(
        x,
        weight if binding is None else binding.parameters[0],
        binding,
        records,
        kind,
        delay,
    )


def run(xs, weight, records, *, recompute=False, kind='direct', delay=0, direct=False):
    region = WeightGradientAccumulator('accumulator_test', (weight,), direct_parameter_grads=direct)

    def compute(x):
        with region.compute():
            return fragment(x, weight, records, kind, delay)

    with region:
        outputs = [checkpoint(compute, x, use_reentrant=False) if recompute else compute(x) for x in xs]
    return outputs, region


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('recompute', [False, True])
def test_direct_leaf_grads_accumulate_across_microbatches_and_reset(device, recompute):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    from nnscaler.runtime.stream import register_parameter_stream
    w = torch.nn.Parameter(torch.randn(4, 6, device=device))
    register_parameter_stream(w, 'accumulator_test')
    ref = w.detach().clone().requires_grad_()
    for step in range(3):
        w.grad = ref.grad = None
        for micro in range(3):
            x = torch.randn(5, 4, device=device, requires_grad=True)
            rx = x.detach().clone().requires_grad_()
            records = []
            ys, region = run([x[:2], x[2:]], w, records, recompute=recompute, direct=True)
            (sum(y.square().sum() for y in ys) + .3 * w.square().sum()).backward()
            ((rx @ ref).square().sum() + .3 * ref.square().sum()).backward()
            torch.testing.assert_close(w.grad, ref.grad)
            torch.testing.assert_close(x.grad, rx.grad)
            assert records == [w.grad.data_ptr()] * 2
            assert not region.tasks


@pytest.mark.parametrize('hooked', [False, True])
def test_direct_leaf_grads_preserve_partial_backward_and_tensor_hooks(hooked):
    w = torch.nn.Parameter(torch.randn(4, 6))
    x = torch.randn(5, 4, requires_grad=True)
    seen = []
    if hooked:
        w.register_hook(lambda grad: seen.append(grad.clone()))
    ys, region = run([x], w, [], direct=True)
    expected = x.detach().t() @ torch.ones_like(ys[0])
    dw, = torch.autograd.grad(ys[0].sum(), (w,), retain_graph=True)
    torch.testing.assert_close(dw, expected)
    assert w.grad is None
    dx, = torch.autograd.grad(ys[0].sum(), (x,), retain_graph=True)
    torch.testing.assert_close(dx, torch.ones_like(ys[0]) @ w.detach().t())
    assert w.grad is None
    ys[0].sum().backward()
    torch.testing.assert_close(w.grad, expected)
    assert len(seen) == (2 if hooked else 0)
    assert not region.tasks


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_direct_grads_survive_caller_reads_after_zero_grad():
    from nnscaler.runtime.stream import register_parameter_stream
    w = torch.nn.Parameter(torch.randn(128, 128, device='cuda'))
    register_parameter_stream(w, 'accumulator_test')
    x = torch.randn(256, 128, device='cuda', requires_grad=True)
    outputs, region = run([x], w, [], direct=True)
    outputs[0].sum().backward()
    expected = w.grad.cpu()
    torch.cuda._sleep(50_000_000)
    read = w.grad.clone()
    w.grad = None
    del outputs
    with torch.cuda.stream(region.stream):
        scratch = [torch.empty_like(w).fill_(float('nan')) for _ in range(32)]
    torch.testing.assert_close(read.cpu(), expected)


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
            assert len(set(records[3 * iteration : 3 * (iteration + 1)])) == 1
    assert len(hooks) == 2


def test_pruned_parameter_edges_do_not_accumulate_and_release_the_region():
    x = torch.randn(4, 4, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(4, 6))
    actual, region = run([x], weight, [])
    (dx,) = torch.autograd.grad(actual[0].sum(), (x,))
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


@pytest.mark.parametrize(
    'kind,match',
    [('failure', 'backend failure'), ('replace', 'preserve'), ('shape', 'preserve')],
)
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


def test_higher_order_backward_fails_explicitly():
    x = torch.randn(4, 4, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(4, 6))
    actual, region = run([x], weight, [])
    with pytest.raises(RuntimeError, match='first-order'):
        torch.autograd.grad(actual[0].sum(), (x, weight), create_graph=True)
    assert not region.tasks


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
            (sum(y.square().sum() for y in outputs) + weight.square().sum() * 0.3).backward()
            ((rx @ ref_weight).square().sum() + ref_weight.square().sum() * 0.3).backward()
            offset = bucket._pofset[weight]
            actual = bucket._contiguous_grads[offset : offset + weight.numel()].view_as(weight)
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
    expected = sum(x.detach().float().t() @ torch.ones_like(y, dtype=torch.float32) for x, y in zip(xs, ys))
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
    assert dx_ready.elapsed_time(dw_ready) > 1.0, 'dInput was blocked behind dWeight'
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
    expected = sum(value.detach().sin().t() @ torch.ones_like(output) for value, output in zip(originals, outputs))
    torch.testing.assert_close(weight.grad, expected)
    assert len(set(records)) == 1 and not region.tasks
