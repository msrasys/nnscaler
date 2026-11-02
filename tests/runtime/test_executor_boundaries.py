# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import gc
import weakref

import pytest
import torch

from nnscaler.runtime.executor import AsyncCommHandler, Executor


@pytest.fixture(autouse=True)
def clear_boundaries():
    Executor.clear()
    AsyncCommHandler().clear()
    yield
    Executor.clear()
    AsyncCommHandler().clear()


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('repeats', [1, 2, 3])
def test_retained_repeated_inputs_accumulate_once_per_segment(split, repeats):
    leaf = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    boundary = leaf * 2
    boundary.retain_grad()
    reference = boundary.detach().requires_grad_()
    weight = torch.nn.Parameter(torch.randn(3, 4, dtype=torch.float64))

    def segment(*inputs):
        return sum((i + 1) * (tensor @ weight) for i, tensor in enumerate(inputs))

    for _ in range(2):
        segment(*([reference] * repeats)).sum().backward()
        output = Executor.fexecute('segment', segment, *([boundary] * repeats))
        if split:
            result = Executor.backward_input('segment', [boundary] * repeats,
                                             [output], [torch.ones_like(output)], [weight])
            Executor.backward_weight('segment', [weight])
        else:
            result = Executor.backward('segment', [boundary] * repeats,
                                       [output], [torch.ones_like(output)])
        expected = sum(range(1, repeats + 1)) * (torch.ones_like(output) @ weight.detach().T)
        for grad in (result,) if repeats == 1 else result:
            torch.testing.assert_close(grad, expected)
        torch.testing.assert_close(boundary.grad, reference.grad)
        assert leaf.grad is None
    Executor.check_clear()


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('repeats', [1, 2])
@pytest.mark.parametrize('replacement_requires_grad', [False, True])
def test_async_replacement_preserves_callback_and_input_identity(
    monkeypatch, split, repeats, replacement_requires_grad,
):
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    leaf = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    boundary = leaf * 2
    boundary.retain_grad()
    replacement = boundary.detach().clone().requires_grad_(replacement_requires_grad)
    weight = torch.nn.Parameter(torch.randn(3, 4, dtype=torch.float64))
    callbacks, waits = [], []

    class Work:
        def wait(self):
            waits.append(True)

    def resume(grad):
        callbacks.append(grad)
        boundary.backward(grad)

    Executor.register_input_grad_callback(boundary, resume)
    AsyncCommHandler().submit(boundary, [Work()], lambda _: replacement)
    inputs = [boundary] * repeats
    output = Executor.fexecute('segment', lambda *xs: sum(x @ weight for x in xs), *inputs)
    expected = repeats * (torch.ones_like(output) @ weight.detach().T)
    if split:
        result = Executor.backward_input('segment', inputs, [output], [torch.ones_like(output)], [weight])
        Executor.backward_weight('segment', [weight])
    else:
        result = Executor.backward('segment', inputs, [output], [torch.ones_like(output)])
    assert len(waits) == len(callbacks) == 1
    for grad in (result,) if repeats == 1 else result:
        torch.testing.assert_close(grad, expected)
    torch.testing.assert_close(boundary.grad, expected)
    torch.testing.assert_close(leaf.grad, expected * 2)
    torch.testing.assert_close(weight.grad, repeats * (boundary.detach().T @ torch.ones_like(output)))
    Executor.check_clear()
    AsyncCommHandler().check_clear()


@pytest.mark.parametrize('split', [False, True])
def test_callback_owns_retained_boundary_gradient(split):
    leaf = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    boundary = leaf * 2
    boundary.retain_grad()
    weight = torch.nn.Parameter(torch.randn(3, 4, dtype=torch.float64))
    calls = []

    def callback(grad):
        calls.append(grad)
        boundary.backward(grad)

    Executor.register_input_grad_callback(boundary, callback)
    output = Executor.fexecute('segment', lambda tensor: tensor @ weight, boundary)
    expected = torch.ones_like(output) @ weight.detach().T
    if split:
        Executor.backward_input('segment', [boundary], [output], [torch.ones_like(output)], [weight])
        Executor.backward_weight('segment', [weight])
    else:
        Executor.backward('segment', [boundary], [output], [torch.ones_like(output)])
    assert len(calls) == 1
    torch.testing.assert_close(boundary.grad, expected)
    torch.testing.assert_close(leaf.grad, expected * 2)
    Executor.check_clear()


@pytest.mark.parametrize('requires_grad', [False, True])
def test_adapter_waits_before_consuming_async_result(requires_grad):
    placeholder = torch.zeros(2)
    completed = torch.tensor([3., 5.])
    waits = []

    class Work:
        def wait(self):
            waits.append(True)

    AsyncCommHandler().submit(placeholder, [Work()], lambda _: completed)
    result = Executor.aexecute(lambda x, y: x + y, placeholder, placeholder,
                               requires_grad=requires_grad)
    torch.testing.assert_close(result, completed * 2)
    assert waits == [True]
    AsyncCommHandler().check_clear()


@pytest.mark.parametrize('split', [False, True])
def test_forward_only_reference_inputs_are_not_pinned_until_backward(split):
    student = torch.randn(4, requires_grad=True)
    reference = torch.randn(1024)
    reference_ref = weakref.ref(reference)
    metadata = {'reference': reference}

    def segment(x, ref, info):
        with torch.no_grad():
            target = ref.square().sum() + info['reference'].sum()
        return x.square() + target

    output = Executor.fexecute('segment', segment, student, reference, metadata)
    del reference, metadata
    gc.collect()
    assert reference_ref() is None
    if split:
        actual = Executor.backward_input('segment', [student], [output], [torch.ones_like(output)], [])
        Executor.backward_weight('segment', [])
    else:
        actual = Executor.backward('segment', [student], [output], [torch.ones_like(output)])
    torch.testing.assert_close(actual, 2 * student)
    Executor.check_clear()


@pytest.mark.parametrize('checkpoint', [False, True])
def test_autograd_keeps_constant_inputs_needed_for_weight_gradients(checkpoint):
    import torch.utils.checkpoint as ckpt

    constant = torch.randn(3, 4)
    expected = constant.clone().sum(0).reshape(4, 1).expand(4, 2)
    weight = torch.nn.Parameter(torch.randn(4, 2))

    def segment(x):
        if checkpoint:
            return ckpt.checkpoint(lambda value: value @ weight, x, use_reentrant=False)
        return x @ weight

    output = Executor.fexecute('segment', segment, constant)
    assert Executor._detach['segment'][0][0][1] is None
    del constant
    gc.collect()
    Executor.backward('segment', [], [output], [torch.ones_like(output)])
    torch.testing.assert_close(weight.grad, expected)
    Executor.check_clear()


def test_grad_enabled_inside_forward_still_has_an_input_gradient():
    value = torch.randn(4)

    def segment(x):
        x.requires_grad_()
        return x.square()

    output = Executor.fexecute('segment', segment, value)
    result = Executor.backward('segment', [value], [output], [torch.ones_like(output)])
    torch.testing.assert_close(result, 2 * value)
    Executor.check_clear()


def test_pruned_input_id_cannot_consume_a_later_boundary_callback():
    later = torch.randn(4, requires_grad=True)
    received = []
    Executor.register_input_grad_callback(later, received.append)
    Executor._run_input_grad_callbacks([(id(later), None)], {})
    assert id(later) in Executor._input_grad_callbacks
    assert not received
    output = Executor.fexecute('later', lambda x: x.square(), later)
    Executor.backward('later', [later], [output], [torch.ones_like(output)])
    assert len(received) == 1
    torch.testing.assert_close(received[0], 2 * later)
    Executor.check_clear()


@pytest.mark.parametrize('split', [False, True])
def test_enabling_input_grad_after_forward_is_rejected(split):
    value = torch.randn(4)
    weight = torch.nn.Parameter(torch.ones(4))
    output = Executor.fexecute('segment', lambda x: x * weight, value)
    value.requires_grad_()
    with pytest.raises(RuntimeError, match='requires_grad.*after forward'):
        if split:
            Executor.backward_input('segment', [value], [output], [torch.ones_like(output)], [weight])
        else:
            Executor.backward('segment', [value], [output], [torch.ones_like(output)])
