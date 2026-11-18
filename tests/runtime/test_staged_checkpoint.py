import gc
import weakref

import pytest
import torch

from nnscaler.runtime.staged_checkpoint import StagedCheckpoint


def functions():
    def prefix(x, w, v, metadata):
        y = torch.nn.functional.dropout(x @ w, p=.2, training=True)
        return {'y': y, 'v': v, 'metadata': metadata}
    def tail(values):
        y = torch.nn.functional.dropout(values['y'].sin() @ values['v'], p=.1, training=True)
        return {'y': y * values['metadata'].numel(), 'meta': (12, None)}
    def whole(x, w, v, metadata):
        return tail(prefix(x, w, v, metadata))
    return whole, prefix, tail


def point(prefix, tail, limit=1 << 20):
    state = StagedCheckpoint('staged_test', limit)
    state.set_stages(prefix, tail, (0, 1))
    return state


def test_exact_rng_parameter_gradients_and_tail_metadata_changes():
    whole, prefix, tail = functions()
    x = torch.randn(11, 7, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(7, 13))
    v = torch.nn.Parameter(torch.randn(13, 17))
    for iteration, size in enumerate((2, 5, 9)):
        metadata = torch.zeros(size)
        torch.manual_seed(68)
        reference = whole(x, w, v, metadata)
        expected = torch.autograd.grad(reference['y'].sum(), (x, w, v))
        rng = torch.get_rng_state()
        torch.manual_seed(68)
        state = point(prefix, tail)
        actual = state.checkpoint(whole, x, w, v, metadata)
        assert torch.equal(torch.get_rng_state(), rng)
        assert actual['meta'] == reference['meta']
        gradients = torch.autograd.grad(actual['y'].sum(), (x, w, v))
        assert torch.equal(torch.get_rng_state(), rng)
        for a, b in zip((actual['y'], *gradients), (reference['y'], *expected)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert state.stats['launched'] == int(iteration > 0)
        assert state.stats['reused'] == state.stats['launched']
        assert state.budget.used == 0


def test_mutation_after_prefetch_rejected_and_budget_released():
    whole, prefix, tail = functions()
    x = torch.randn(11, 7, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(7, 13))
    v = torch.nn.Parameter(torch.randn(13, 17))
    for iteration in range(2):
        state = point(prefix, tail)
        result = state.checkpoint(whole, x, w, v, torch.zeros(2))
        if iteration:
            with torch.no_grad():
                x.add_(1)
            with pytest.raises(RuntimeError, match='modified by an inplace operation'):
                result['y'].sum().backward()
        else:
            result['y'].sum().backward()
        assert state.budget.used == 0


def test_budget_fallback_abandoned_forward_and_repeated_backward():
    whole, prefix, tail = functions()
    x = torch.randn(11, 7, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(7, 13))
    v = torch.nn.Parameter(torch.randn(13, 17))
    args = (x, w, v, torch.zeros(2))
    state = point(prefix, tail)
    result = state.checkpoint(whole, *args)
    torch.autograd.grad(result['y'].sum(), (x, w, v))
    state = point(prefix, tail, limit=1)
    result = state.checkpoint(whole, *args)
    assert state.stats['budget_skips'] == 1
    first = torch.autograd.grad(result['y'].sum(), (x, w, v), retain_graph=True)
    second = torch.autograd.grad(result['y'].sum(), (x, w, v))
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert state.budget.used == 0
    state = point(prefix, tail)
    result = state.checkpoint(whole, *args)
    budget = state.budget
    assert budget.used > 0
    reference = weakref.ref(state)
    del result, state
    gc.collect()
    assert reference() is None and budget.used == 0


def test_nondifferentiable_outputs_and_no_grad():
    def prefix(x):
        return {'x': x.square()}
    def tail(data):
        return data['x'].sin(), data['x'].detach()
    def whole(x):
        return tail(prefix(x))
    x = torch.randn(7, requires_grad=True)
    state = StagedCheckpoint('flags', 1 << 20)
    state.set_stages(prefix, tail, (0,), (True, False))
    y, detached = state.checkpoint(whole, x)
    assert y.requires_grad and not detached.requires_grad
    torch.autograd.grad(y.sum(), (x,))
    with torch.no_grad():
        state = StagedCheckpoint('flags', 1 << 20)
        state.set_stages(prefix, tail)
        actual = state.checkpoint(whole, x)
        expected = whole(x)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_storage_growth_is_rejected_and_releases_budget():
    repeat = [1]
    def prefix(x):
        return {'y': x.repeat(repeat[0], 1).sin()}
    def tail(data):
        return data['y'].sum()
    def whole(x):
        return tail(prefix(x))
    x = torch.randn(11, 7, requires_grad=True)
    state = StagedCheckpoint('growth', 1 << 20)
    state.set_stages(prefix, tail)
    state.checkpoint(whole, x).backward()
    repeat[0] = 2
    state = StagedCheckpoint('growth', 1 << 20)
    state.set_stages(prefix, tail)
    with pytest.raises(RuntimeError, match='storage grew'):
        state.checkpoint(whole, x)
    assert state.budget.used == 0


def test_abandoned_graph_releases_budget_while_launch_point_is_retained():
    whole, prefix, tail = functions()
    x = torch.randn(11, 7, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(7, 13))
    v = torch.nn.Parameter(torch.randn(13, 17))
    args = (x, w, v, torch.zeros(2))
    warm = point(prefix, tail)
    warm.checkpoint(whole, *args)['y'].sum().backward()
    state = point(prefix, tail)
    y = state.checkpoint(whole, *args)
    assert state.budget.used > 0
    del y
    gc.collect()
    assert state.budget.used == 0 and state.prefix_values is None


def test_concurrent_backward_is_rejected_without_clearing_active_work():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    entered, finish = Event(), Event()
    x = torch.randn(7, requires_grad=True)
    def prefix(x):
        return {'y': x.sin()}
    def tail(data):
        entered.set()
        if not finish.wait(5):
            raise RuntimeError('Test timed out waiting for concurrent backward')
        return data['y'].square()
    def whole(x):
        return x.sin().square()
    state = StagedCheckpoint('concurrent', 1 << 20)
    state.set_stages(prefix, tail)
    y = state.checkpoint(whole, x)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(torch.autograd.grad, y.sum(), (x,), retain_graph=True)
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match='Concurrent backward'):
                torch.autograd.grad(y.sum(), (x,), retain_graph=True)
        finally:
            finish.set()
        result, = future.result(timeout=5)
    torch.testing.assert_close(result, 2 * x.sin() * x.cos(), rtol=0, atol=0)


def test_outer_nonreentrant_checkpoint_rejected():
    from torch.utils.checkpoint import checkpoint
    def prefix(x):
        return {'x': x.sin()}
    def tail(data):
        return data['x'].square()
    def whole(x):
        return tail(prefix(x))
    state = StagedCheckpoint('nested', 1 << 20)
    state.set_stages(prefix, tail)
    x = torch.randn(7, requires_grad=True)
    with pytest.raises(RuntimeError, match='nested checkpoints'):
        checkpoint(lambda value: state.checkpoint(whole, value), x, use_reentrant=False)
