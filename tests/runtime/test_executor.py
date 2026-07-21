#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch

from nnscaler.runtime import executor as executor_module
from nnscaler.runtime.executor import AsyncCommHandler, Executor


def test_backward_with_pseudo_freed_output_tensor():
    Executor.clear()
    try:
        torch.manual_seed(0)
        x = torch.randn(4, 3, requires_grad=True)
        grad = torch.randn(4, 3)

        def segment(inp):
            return torch.sin(inp * 2.0)

        baseline = torch.autograd.grad(segment(x), x, grad)[0]
        output = Executor.fexecute('segment', segment, x)

        Executor.pseudo_free_tensor(output)
        assert tuple(output.shape) == (1,)

        actual = Executor.backward('segment', (x,), (output,), (grad,))
        assert torch.allclose(actual, baseline)
        Executor.check_clear()
    finally:
        Executor.clear()


def test_backward_with_mixed_tensor_and_gradient_edge_roots():
    Executor.clear()
    try:
        torch.manual_seed(0)
        x = torch.randn(4, 3, requires_grad=True)
        grads = (torch.randn(4, 3), torch.randn(4, 3))

        def segment(inp):
            return torch.sin(inp), torch.cos(inp)

        baseline = torch.autograd.grad(segment(x), x, grads)[0]
        outputs = Executor.fexecute('segment', segment, x)

        Executor.pseudo_free_tensor(outputs[0])
        assert tuple(outputs[0].shape) == (1,)
        assert tuple(outputs[1].shape) == (4, 3)

        actual = Executor.backward('segment', (x,), outputs, grads)
        assert torch.allclose(actual, baseline)
        Executor.check_clear()
    finally:
        Executor.clear()


def test_deferred_pseudo_free_waits_for_all_sends():
    Executor.clear()
    try:
        torch.manual_seed(0)
        x = torch.randn(4, 3, requires_grad=True)
        grad = torch.randn(4, 3)

        def segment(inp):
            return torch.cos(inp * 3.0)

        baseline = torch.autograd.grad(segment(x), x, grad)[0]
        output = Executor.fexecute('segment', segment, x)

        Executor.defer_pseudo_free_tensor(output)
        Executor.defer_pseudo_free_tensor(output)
        Executor.complete_deferred_pseudo_free_tensor(output)
        assert tuple(output.shape) == (4, 3)
        Executor.complete_deferred_pseudo_free_tensor(output)
        assert tuple(output.shape) == (1,)

        actual = Executor.backward('segment', (x,), (output,), (grad,))
        assert torch.allclose(actual, baseline)
        Executor.check_clear()
    finally:
        Executor.clear()


def test_deferred_pseudo_free_skips_if_backward_already_consumed_edge():
    Executor.clear()
    try:
        torch.manual_seed(0)
        x = torch.randn(4, 3, requires_grad=True)
        grad = torch.randn(4, 3)

        def segment(inp):
            return torch.tanh(inp)

        baseline = torch.autograd.grad(segment(x), x, grad)[0]
        output = Executor.fexecute('segment', segment, x)

        Executor.defer_pseudo_free_tensor(output)
        actual = Executor.backward('segment', (x,), (output,), (grad,))
        Executor.complete_deferred_pseudo_free_tensor(output)

        assert tuple(output.shape) == (4, 3)
        assert torch.allclose(actual, baseline)
        Executor.check_clear()
    finally:
        Executor.clear()


def test_pseudo_free_skips_leaf_tensors():
    Executor.clear()
    try:
        tensor = torch.randn(4, 3, requires_grad=True)
        Executor.pseudo_free_tensor(tensor)
        assert tuple(tensor.shape) == (4, 3)
        Executor.check_clear()
    finally:
        Executor.clear()


def test_pseudo_free_skips_view_tensors():
    Executor.clear()
    try:
        x = torch.randn(4, 3, requires_grad=True)
        output = (x * 2.0).view(3, 4)

        Executor.pseudo_free_tensor(output)

        assert tuple(output.shape) == (3, 4)
        Executor.check_clear()
    finally:
        Executor.clear()


def test_deferred_pseudo_free_records_gradient_edge_once(monkeypatch):
    Executor.clear()
    try:
        x = torch.randn(4, 3, requires_grad=True)
        output = torch.sin(x)
        calls = {'count': 0}
        original = executor_module.get_gradient_edge

        def counted_get_gradient_edge(tensor):
            calls['count'] += 1
            return original(tensor)

        monkeypatch.setattr(
            executor_module,
            'get_gradient_edge',
            counted_get_gradient_edge,
        )
        Executor.defer_pseudo_free_tensor(output)
        Executor.complete_deferred_pseudo_free_tensor(output)

        assert calls['count'] == 1
        Executor._pseudo_free_grad_edges.clear()
        Executor.check_clear()
    finally:
        Executor.clear()


def test_pseudo_free_warns_when_gradient_edge_is_unavailable(monkeypatch, caplog):
    Executor.clear()
    Executor._pseudo_free_unavailable_warned = False
    try:
        output = torch.sin(torch.randn(4, 3, requires_grad=True))
        monkeypatch.setattr(executor_module, 'get_gradient_edge', None)

        Executor.defer_pseudo_free_tensor(output)
        Executor.defer_pseudo_free_tensor(output)

        assert tuple(output.shape) == (4, 3)
        assert caplog.text.count('get_gradient_edge') == 1
        Executor.check_clear()
    finally:
        Executor._pseudo_free_unavailable_warned = False
        Executor.clear()


def test_async_send_callback_runs_when_drained():
    class Work:
        def __init__(self):
            self.waited = False

        def wait(self):
            self.waited = True

        def is_completed(self):
            return True

    AsyncCommHandler().clear()
    try:
        work = Work()
        called = {'value': False}
        AsyncCommHandler().hold_send(
            torch.empty(1),
            work,
            callback=lambda: called.__setitem__('value', True),
        )
        AsyncCommHandler().drain_sends(wait=False)

        assert work.waited
        assert called['value']
        AsyncCommHandler().check_clear()
    finally:
        AsyncCommHandler().clear()


def test_send_bundle_waits_for_oldest_bundle_at_window_limit():
    class Work:
        def __init__(self):
            self.waited = False

        def wait(self):
            self.waited = True

    handler = AsyncCommHandler()
    handler.clear()
    try:
        works = [Work() for _ in range(3)]
        callbacks = [False, False, False]

        for index in range(2):
            AsyncCommHandler().reserve_send_bundle(((0, 1),))
            AsyncCommHandler().begin_send_bundle(((0, 1),))
            AsyncCommHandler().hold_send(
                torch.empty(1),
                works[index],
                callback=lambda index=index: callbacks.__setitem__(index, True),
            )
            AsyncCommHandler().end_send_bundle()

        AsyncCommHandler().reserve_send_bundle(((0, 1),))
        assert works[0].waited
        assert callbacks == [True, False, False]
        AsyncCommHandler().begin_send_bundle(((0, 1),))
        AsyncCommHandler().hold_send(
            torch.empty(1),
            works[2],
            callback=lambda: callbacks.__setitem__(2, True),
        )
        AsyncCommHandler().end_send_bundle()

        handler.drain()
        assert all(work.waited for work in works)
        assert callbacks == [True, True, True]
        handler.check_clear()
    finally:
        handler.clear()
