#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch

import nnscaler.runtime.executor as executor
from nnscaler.parallel import ComputeConfig
from nnscaler.runtime import _patch_torch
from nnscaler.runtime.executor import Executor


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


@pytest.mark.parametrize('requires_grad', [False, True])
def test_adapter_waits_for_async_input(monkeypatch, requires_grad):
    handler = executor._AsyncCommHandler()
    monkeypatch.setattr(executor, '_instance', handler)
    pending = torch.zeros(8)
    events = []

    class Work:
        def is_completed(self):
            return False

        def wait(self):
            events.append('wait')
            pending.copy_(torch.arange(8, dtype=torch.float32))

    # Exercise the existing callback path too: the actual input has a
    # different shape from the tensor carrying the communication handle.
    handler.submit(pending, [Work()], lambda tensor: tensor[:4])

    def adapter(tensor):
        events.append('consume')
        return tensor * 2

    result = Executor.aexecute(adapter, pending, requires_grad=requires_grad)
    assert events == ['wait', 'consume']
    torch.testing.assert_close(result, torch.arange(4, dtype=torch.float32) * 2)
    assert result.requires_grad == requires_grad
    handler.check_clear()


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


def test_custom_fbw_restores_module_symbols():
    original_input = executor.backward_input
    original_weight = executor.backward_weight

    def outer_input(*args, **kwargs):
        pass

    def outer_weight(*args, **kwargs):
        pass

    def inner_input(*args, **kwargs):
        pass

    def inner_weight(*args, **kwargs):
        pass

    with pytest.raises(RuntimeError, match='expected'):
        with executor.custom_fbw(outer_input, outer_weight):
            assert executor.backward_input is outer_input
            assert executor.backward_weight is outer_weight
            with executor.custom_fbw(inner_input, inner_weight):
                assert executor.backward_input is inner_input
                assert executor.backward_weight is inner_weight
            assert executor.backward_input is outer_input
            assert executor.backward_weight is outer_weight
            raise RuntimeError('expected')

    assert executor.backward_input is original_input
    assert executor.backward_weight is original_weight


def test_custom_fbw_config_does_not_require_default_backend(monkeypatch):
    monkeypatch.setattr(_patch_torch, 'FBW_SUPPORTED', False)

    config = ComputeConfig(
        plan_ngpus=1,
        runtime_ngpus=1,
        use_end2end=True,
        use_fbw=True,
    )

    assert config.use_fbw


def test_custom_fbw_grad_coverage_accepts_split_gradients():
    stage_input = torch.randn(2, 4, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(4, 4))
    output = stage_input @ weight
    output_grad = torch.ones_like(output)

    def backward_input(name, input_tensors, output_tensors, output_grads, weights):
        torch.autograd.backward(
            output_tensors,
            grad_tensors=output_grads,
            inputs=input_tensors,
            retain_graph=True,
        )
        return input_tensors[0].grad

    def backward_weight(name, weights):
        weight_grad = stage_input.detach().T @ output_grad
        torch.autograd.backward((weight,), grad_tensors=(weight_grad,))

    with executor.custom_fbw(
        backward_input,
        backward_weight,
        check_grad_coverage=True,
    ):
        executor.backward_input(
            'segment', [stage_input], [output], [output_grad], (weight,)
        )
        executor.backward_weight('segment', (weight,))
