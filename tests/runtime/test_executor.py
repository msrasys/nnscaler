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


@pytest.mark.parametrize('repeats', [1, 2, 3])
def test_backward_retains_repeated_boundary_input_once(repeats):
    leaf = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    outer_input = leaf * 2
    outer_input.retain_grad()
    reference = outer_input.detach().requires_grad_()

    def segment(*inputs):
        return sum((index + 1) * tensor.sin() for index, tensor in enumerate(inputs))

    # Separate segments accumulate, while repeated arguments within one segment
    # must not multiply the already-accumulated detached input gradient.
    for step in range(2):
        expected = sum(range(1, repeats + 1)) * reference.cos()
        segment(*([reference] * repeats)).sum().backward()
        output = Executor.fexecute('segment', segment, *([outer_input] * repeats))
        grads = Executor.backward(
            'segment', [outer_input] * repeats, [output], [torch.ones_like(output)],
        )
        for grad in (grads,) if repeats == 1 else grads:
            torch.testing.assert_close(grad, expected)
        torch.testing.assert_close(outer_input.grad, reference.grad)
        assert leaf.grad is None
    Executor.check_clear()


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


@pytest.mark.parametrize('retain_grad', [False, True])
def test_input_grad_callback_resumes_outer_autograd_once(retain_grad):
    Executor.clear()
    try:
        weight = torch.nn.Parameter(torch.tensor(2.0))
        outer_input = weight * 3
        if retain_grad:
            outer_input.retain_grad()
        output = Executor.fexecute(
            'segment', lambda tensor, _metadata: torch.sin(tensor),
            outer_input, {'kind': 'non_tensor_input'},
        )
        callback_grads = []

        def resume_outer_backward(grad):
            callback_grads.append(grad)
            torch.autograd.backward(outer_input, grad)

        Executor.register_input_grad_callback(outer_input, resume_outer_backward)
        actual = Executor.backward(
            'segment', (outer_input,), (output,), (torch.ones_like(output),),
        )

        expected = torch.cos(outer_input)
        assert torch.equal(actual, expected)
        assert len(callback_grads) == 1
        assert torch.equal(callback_grads[0], expected)
        assert torch.equal(weight.grad, expected * 3)
        if retain_grad:
            torch.testing.assert_close(outer_input.grad, expected)
        Executor.check_clear()
    finally:
        Executor.clear()



def test_split_backward_runs_input_grad_callback():
    weight = torch.nn.Parameter(torch.tensor(2.0))
    outer_input = weight * 3
    segment_input = outer_input.reshape(1, 1)
    linear = torch.nn.Linear(1, 1, bias=False)
    output = Executor.fexecute('segment', linear, segment_input)

    def resume_outer_backward(grad):
        torch.autograd.backward(outer_input, grad.reshape(()))

    Executor.register_input_grad_callback(segment_input, resume_outer_backward)
    Executor.backward_input(
        'segment', [segment_input], [output],
        [torch.ones_like(output)], linear.parameters(),
    )

    assert weight.grad is not None
    Executor.backward_weight('segment', linear.parameters())
    Executor.check_clear()


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('completed', [False, True])
@pytest.mark.parametrize('repeats', [1, 2])
def test_input_grad_callback_survives_async_replacement(monkeypatch, split, completed, repeats):
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    leaf = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    boundary = leaf * 2
    replacement = boundary.detach().clone().requires_grad_()
    weight = torch.nn.Parameter(torch.randn(3, 4, dtype=torch.float64))
    callbacks = []

    class Work:
        def is_completed(self):
            return completed

        def wait(self):
            pass

    def resume(grad):
        callbacks.append(grad)
        boundary.backward(grad)

    Executor.register_input_grad_callback(boundary, resume)
    executor.AsyncCommHandler().submit(boundary, [Work()], lambda tensor: replacement)
    inputs = [boundary] * repeats
    output = Executor.fexecute('segment', lambda *xs: sum(x @ weight for x in xs), *inputs)
    expected = repeats * (torch.ones_like(output) @ weight.detach().T)
    if split:
        grad = Executor.backward_input(
            'segment', inputs, [output], [torch.ones_like(output)], [weight],
        )
        Executor.backward_weight('segment', [weight])
    else:
        grad = Executor.backward('segment', inputs, [output], [torch.ones_like(output)])
    assert len(callbacks) == 1
    for value in (grad,) if repeats == 1 else grad:
        torch.testing.assert_close(value, expected)
    torch.testing.assert_close(leaf.grad, expected * 2)
    torch.testing.assert_close(weight.grad, repeats * (boundary.detach().T @ torch.ones_like(output)))
    Executor.check_clear()
    executor.AsyncCommHandler().check_clear()


@pytest.mark.parametrize('check_coverage', [False, True])
@pytest.mark.parametrize('use_default', [False, True])
def test_custom_fbw_dispatches_input_grad_callback(check_coverage, use_default):
    leaf = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    boundary = leaf * 2
    boundary.retain_grad()
    weight = torch.nn.Parameter(torch.randn(3, 4, dtype=torch.float64))
    reference_weight = weight.detach().clone().requires_grad_()
    reference_input = boundary.detach().clone().requires_grad_()
    (reference_input @ reference_weight).sum().backward()
    output = Executor.fexecute('segment', lambda tensor: tensor @ weight, boundary)
    callbacks = []
    pending = {}

    def custom_input(name, inputs, outputs, grads, weights):
        pairs = Executor._detach[name].pop(0)
        detached_inputs = [tensor for _, tensor in pairs if tensor.requires_grad]
        result = torch.autograd.grad(outputs, detached_inputs, grads, retain_graph=True)
        pending[name] = (outputs, grads)
        return result[0] if len(result) == 1 else result

    def custom_weight(name, weights):
        outputs, grads = pending.pop(name)
        torch.autograd.backward(outputs, grads, inputs=weights)

    def resume(grad):
        callbacks.append(grad)
        boundary.backward(grad)

    with executor.custom_fbw(
        Executor.backward_input if use_default else custom_input,
        Executor.backward_weight if use_default else custom_weight,
        check_grad_coverage=check_coverage,
    ):
        # Registration happens inside the context, as in a generated schedule.
        Executor.register_input_grad_callback(boundary, resume)
        grad = executor.backward_input(
            'segment', [boundary], [output], [torch.ones_like(output)], [weight],
        )
        assert len(callbacks) == 1
        assert weight.grad is None
        executor.backward_weight('segment', [weight])
    torch.testing.assert_close(grad, reference_input.grad)
    torch.testing.assert_close(boundary.grad, reference_input.grad)
    torch.testing.assert_close(leaf.grad, reference_input.grad * 2)
    torch.testing.assert_close(weight.grad, reference_weight.grad)
    Executor.check_clear()


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
            outer_wrapper = executor.backward_input
            assert outer_wrapper.__wrapped__ is outer_input
            assert executor.backward_weight is outer_weight
            with executor.custom_fbw(inner_input, inner_weight):
                assert executor.backward_input.__wrapped__ is inner_input
                assert executor.backward_weight is inner_weight
            assert executor.backward_input is outer_wrapper
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
