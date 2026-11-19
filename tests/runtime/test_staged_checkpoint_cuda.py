import gc
import weakref

import pytest
import torch

from nnscaler.runtime.cuda_gate import Gate, counts
from nnscaler.runtime.staged_checkpoint import StagedCheckpoint, ReplayReady
from nnscaler.runtime.stream import ModuleStreams, drain_stream_holds
from nnscaler.runtime.deferred import notify_async_launch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def test_early_release_covers_backend_host_count_wait():
    torch.cuda._sleep(1)
    communication = torch.cuda.Stream()
    x = torch.randn(256, 64, device='cuda', requires_grad=True)
    w = torch.nn.Parameter(torch.randn(64, 64, device='cuda'))
    stamps = {}

    def prefix(x, w):
        if torch.is_grad_enabled():
            stamp = torch.cuda.Event(enable_timing=True)
            stamp.record()
            stamps['prefix'] = stamp
        return {'y': x @ w}

    def tail(data):
        return data['y'].sin()

    def whole(x, w):
        return tail(prefix(x, w))

    expected = torch.autograd.grad(whole(x, w).sum(), (x, w))
    warm = StagedCheckpoint('early_count_wait', 1 << 20)
    warm.set_stages(prefix, tail)
    torch.autograd.grad(warm.checkpoint(whole, x, w).sum(), (x, w))
    for early in (False, True):
        with ModuleStreams() as scope:
            state = scope.checkpoint_prefetch('early_count_wait', 1 << 20, staged=True)
            state.set_stages(prefix, tail)
            y = state.checkpoint(whole, x, w)
            if early:
                # The compiler emits this release before the anchor's scope.
                state.prefetch()
            with scope.prefetch_launch_point((state,), submitted=True):
                communication.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(communication):
                    torch.cuda._sleep(40_000_000)
                    counts_done = torch.cuda.Event(enable_timing=True)
                    counts_done.record()
                # Dynamic counts block the host before payload submission.
                counts_done.synchronize()
                notify_async_launch()
        actual = torch.autograd.grad(y.sum(), (x, w))
        torch.cuda.synchronize()
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)
        assert (stamps['prefix'].elapsed_time(counts_done) > 0) == early
        assert state.stats['launched'] == state.stats['reused'] == 1
        assert not state.gate.fallback()
        state.budget.poll()
        assert state.budget.used == 0
    drain_stream_holds(wait=True)


def test_exact_gradients_and_replay_waits_for_anchor_inputs_not_result():
    torch.cuda._sleep(1)  # Warm module loading before any progress gate is closed.
    communication = torch.cuda.Stream()
    stamps = {}

    class Incoming(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.view_as(value)
        @staticmethod
        def backward(ctx, grad):
            caller = torch.cuda.current_stream()
            torch.cuda._sleep(3_000_000)
            ready = torch.cuda.Event(enable_timing=True)
            ready.record()
            stamps['input_ready'] = ready
            stamps['point'].record_ready()
            communication.wait_stream(caller)
            with torch.cuda.stream(communication):
                torch.cuda._sleep(10_000_000)
                done = torch.cuda.Event(enable_timing=True)
                done.record()
            stamps['communication_done'] = done
            caller.wait_event(done)
            return grad

    x = torch.randn(512, 256, device='cuda', requires_grad=True)
    w = torch.nn.Parameter(torch.randn(256, 512, device='cuda'))
    v = torch.nn.Parameter(torch.randn(512, 256, device='cuda'))
    def prefix(x, w, v):
        return {'y': x @ w, 'v': v}
    def tail(data):
        if torch.is_grad_enabled():
            stamp = torch.cuda.Event(enable_timing=True)
            stamp.record()
            stamps['tail_begin'] = stamp
        return data['y'].sin() @ data['v']
    def whole(x, w, v):
        return tail(prefix(x, w, v))
    expected = torch.autograd.grad(whole(x, w, v).sum(), (x, w, v))
    for i in range(3):
        state = StagedCheckpoint('staged_cuda', 1 << 30)
        state.set_stages(prefix, tail)
        state.backward_point = ReplayReady(cuda=True)
        state.backward_point.captured = True
        stamps['point'] = state.backward_point
        y = state.checkpoint(whole, x, w, v)
        state.prefetch()
        state.finish_forward()
        gradients = torch.autograd.grad(Incoming.apply(y).sum(), (x, w, v))
        torch.cuda.synchronize()
        state.budget.poll()
        assert state.budget.used == 0
        for actual, reference in zip(gradients, expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        assert stamps['input_ready'].elapsed_time(stamps['tail_begin']) >= 0
        assert stamps['tail_begin'].elapsed_time(stamps['communication_done']) > 0
        if i:
            assert state.stats['launched'] == state.stats['reused'] == 1
            assert not state.gate.fallback()
    drain_stream_holds(wait=True)


def test_abandoned_forward_and_prefill_exception_drain_and_release_budget():
    fail = [False]
    x = torch.randn(512, 256, device='cuda', requires_grad=True)
    w = torch.nn.Parameter(torch.randn(256, 512, device='cuda'))
    def prefix(x, w):
        y = x @ w
        if fail[0]:
            raise ValueError('injected replay failure')
        return {'y': y}
    def tail(data):
        return data['y'].sin()
    def whole(x, w):
        return (x @ w).sin()
    state = StagedCheckpoint('staged_abandon', 1 << 30)
    state.set_stages(prefix, tail)
    state.checkpoint(whole, x, w).sum().backward()
    state = StagedCheckpoint('staged_abandon', 1 << 30)
    state.set_stages(prefix, tail)
    y = state.checkpoint(whole, x, w)
    state.prefetch()
    state.finish_forward()
    budget = state.budget
    ref = weakref.ref(state)
    del y, state
    gc.collect()
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)
    budget.poll()
    assert ref() is None and budget.used == 0
    state = StagedCheckpoint('staged_abandon', 1 << 30)
    state.set_stages(prefix, tail)
    y = state.checkpoint(whole, x, w)
    state.prefetch()
    state.finish_forward()
    del y
    gc.collect()
    torch.cuda.synchronize()
    state.budget.poll()
    assert state.budget.used == 0 and state.prefix_values is None
    fail[0] = True
    state = StagedCheckpoint('staged_abandon', 1 << 30)
    state.set_stages(prefix, tail)
    with pytest.raises(ValueError, match='injected replay'):
        state.checkpoint(whole, x, w)
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)
    state.budget.poll()
    assert state.budget.used == 0


def test_ready_caller_releases_gate_from_host():
    value = torch.zeros(1, device='cuda')
    value.fill_(19)
    value.zero_()
    torch.cuda.synchronize()
    caller, side = torch.cuda.current_stream(), torch.cuda.Stream()
    gate = Gate(value.device)
    before = counts['host_ready_release']
    with torch.cuda.stream(side):
        gate.wait(side)
        value.fill_(19)
    assert caller.query()
    gate.open(caller)
    caller.wait_stream(side)
    caller.synchronize()
    assert counts['host_ready_release'] == before + 1
    assert value.item() == 19 and not gate.fallback()
    gate.finish()
    drain_stream_holds(wait=True)
