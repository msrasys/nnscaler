# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
