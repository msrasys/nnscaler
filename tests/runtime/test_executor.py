#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
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


def test_backward_preserves_retained_outer_input_grad():
    Executor.clear()
    try:
        leaf = torch.randn(4, 3, requires_grad=True)
        outer_input = leaf * 2
        outer_input.retain_grad()

        output = Executor.fexecute('segment', torch.sin, outer_input)
        expected = torch.cos(outer_input)
        actual = Executor.backward(
            'segment', (outer_input,), (output,), (torch.ones_like(output),),
        )

        assert torch.equal(actual, expected)
        assert torch.equal(outer_input.grad, expected)
        assert leaf.grad is None
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

@pytest.fixture(autouse=True)
def clear_executor():
    Executor.clear()
    yield
    Executor.clear()


def _make_linears(dtype=torch.float64):
    reference = torch.nn.Linear(8, 16, dtype=dtype)
    actual = torch.nn.Linear(8, 16, dtype=dtype)
    actual.load_state_dict(reference.state_dict())
    return reference, actual


def test_split_backward_matches_full_backward():
    torch.manual_seed(0)
    reference, actual = _make_linears()
    input_data = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(4, 16, dtype=torch.float64)

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad)

    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('linear', actual, actual_input)
    input_grad = Executor.backward_input(
        'linear', [actual_input], [output], [output_grad], actual.parameters()
    )

    torch.testing.assert_close(input_grad, reference_input.grad)
    assert actual.weight.grad is None
    assert actual.bias.grad is None

    Executor.backward_weight('linear', actual.parameters())

    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
    Executor.check_clear()


def test_split_backward_uses_fifo_for_multiple_invocations():
    torch.manual_seed(1)
    reference, actual = _make_linears()
    inputs = [
        torch.randn(4, 8, dtype=torch.float64),
        torch.randn(4, 8, dtype=torch.float64),
    ]
    output_grads = [
        torch.randn(4, 16, dtype=torch.float64),
        torch.randn(4, 16, dtype=torch.float64),
    ]

    reference_inputs = [value.clone().requires_grad_() for value in inputs]
    for input_tensor, output_grad in zip(reference_inputs, output_grads):
        reference(input_tensor).backward(output_grad)

    actual_inputs = [value.clone().requires_grad_() for value in inputs]
    outputs = [Executor.fexecute('linear', actual, value) for value in actual_inputs]
    input_grads = [
        Executor.backward_input(
            'linear', [input_tensor], [output], [output_grad], actual.parameters()
        )
        for input_tensor, output, output_grad in zip(actual_inputs, outputs, output_grads)
    ]

    assert actual.weight.grad is None
    assert actual.bias.grad is None
    Executor.backward_weight('linear', actual.parameters())
    Executor.backward_weight('linear', actual.parameters())

    for input_grad, reference_input in zip(input_grads, reference_inputs):
        torch.testing.assert_close(input_grad, reference_input.grad)
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
    Executor.check_clear()


def test_split_backward_without_input_gradient():
    torch.manual_seed(2)
    reference, actual = _make_linears()
    input_tensor = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(4, 16, dtype=torch.float64)

    reference(input_tensor).backward(output_grad)

    output = Executor.fexecute('linear', actual, input_tensor)
    input_grad = Executor.backward_input(
        'linear', [], [output], [output_grad], actual.parameters()
    )

    assert input_grad is None
    assert actual.weight.grad is None
    assert actual.bias.grad is None

    Executor.backward_weight('linear', actual.parameters())

    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
    Executor.check_clear()


def test_split_backward_with_view_output():
    class ViewLinear(torch.nn.Linear):
        def forward(self, input_tensor):
            return super().forward(input_tensor).view(8, 8)

    torch.manual_seed(3)
    reference = ViewLinear(8, 16, dtype=torch.float64)
    actual = ViewLinear(8, 16, dtype=torch.float64)
    actual.load_state_dict(reference.state_dict())
    input_data = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(8, 8, dtype=torch.float64)

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad)

    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('view_linear', actual, actual_input)
    input_grad = Executor.backward_input(
        'view_linear', [actual_input], [output], [output_grad], actual.parameters()
    )
    Executor.backward_weight('view_linear', actual.parameters())

    torch.testing.assert_close(input_grad, reference_input.grad)
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
    Executor.check_clear()


def test_weight_backward_triggers_accumulate_grad_hook():
    module = torch.nn.Linear(8, 16)
    input_tensor = torch.randn(4, 8, requires_grad=True)
    output_grad = torch.randn(4, 16)
    param_tmp = module.weight.expand_as(module.weight)
    grad_acc = param_tmp.grad_fn.next_functions[0][0]
    hook_calls = []
    handle = grad_acc.register_hook(lambda *args: hook_calls.append(args))

    output = Executor.fexecute('linear', module, input_tensor)
    Executor.backward_input(
        'linear', [input_tensor], [output], [output_grad], module.parameters()
    )
    assert hook_calls == []

    Executor.backward_weight('linear', module.parameters())

    assert len(hook_calls) == 1
    handle.remove()
    Executor.check_clear()


def test_split_backward_applies_backward_pre_hook_once():
    torch.manual_seed(4)
    reference, actual = _make_linears()
    input_data = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(4, 16, dtype=torch.float64)
    hook_calls = []

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad * 2)

    def scale_grad(input_tensors, output_tensors, output_grads):
        hook_calls.append(None)
        return input_tensors, output_tensors, [grad * 2 for grad in output_grads]

    Executor.register_backward_pre_hook(scale_grad)
    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('linear', actual, actual_input)
    input_grad = Executor.backward_input(
        'linear', [actual_input], [output], [output_grad], actual.parameters()
    )
    Executor.backward_weight('linear', actual.parameters())

    assert len(hook_calls) == 1
    torch.testing.assert_close(input_grad, reference_input.grad)
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
    Executor.check_clear()


def test_backward_weight_requires_pending_input_backward():
    module = torch.nn.Linear(8, 16)
    with pytest.raises(RuntimeError, match='No pending weight backward'):
        Executor.backward_weight('linear', module.parameters())
