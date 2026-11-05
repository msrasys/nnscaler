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


@pytest.mark.parametrize('use_fbw', [False, True])
def test_backward_rejects_missing_input_gradient(use_fbw):
    class Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
            self.aux_weight = torch.nn.Parameter(torch.tensor(3., dtype=torch.float64))

        def forward(self, x, unused):
            return (x * self.weight).sum(), (unused * self.aux_weight).data

    module = Module()
    x = torch.ones(2, 3, dtype=torch.float64, requires_grad=True)
    unused = torch.ones(4, 2, dtype=torch.float64).t().requires_grad_()
    loss, _ = Executor.fexecute('unused_input', module, x, unused)
    saved_inputs = [tensor for _, tensor in Executor._detach['unused_input'][0]]

    if use_fbw:
        with pytest.raises(AssertionError, match='got gradient None'):
            Executor.backward_input(
                'unused_input', [x, unused], [loss], [None], module.parameters(),
            )
        assert module.weight.grad is None
        Executor.backward_weight('unused_input', module.parameters())
    else:
        with pytest.raises(AssertionError, match='got gradient None'):
            Executor.backward('unused_input', [x, unused], [loss], [None])
        torch.testing.assert_close(saved_inputs[0].grad, torch.full_like(x, 2))

    assert saved_inputs[1].grad is None
    torch.testing.assert_close(module.weight.grad, torch.tensor(6., dtype=torch.float64))
    assert module.aux_weight.grad is None
    assert x.grad is None and unused.grad is None
    Executor.check_clear()


@pytest.mark.parametrize('use_fbw', [False, True])
@pytest.mark.parametrize('ninputs', [0, 1, 2])
def test_backward_without_output_roots_preserves_state_pairing(use_fbw, ninputs):
    weight = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
    inputs = [
        torch.randn(2, idx + 1, dtype=torch.float64, requires_grad=True)
        for idx in range(ninputs)
    ]
    Executor.fexecute(
        'no_roots', lambda *args: tuple((value * weight).data for value in args),
        *inputs,
    )
    if use_fbw:
        grads = Executor.backward_input('no_roots', inputs, [], [], [weight])
        assert len(Executor._weight_backward_states['no_roots']) == 1
        Executor.backward_weight('no_roots', [weight])
    else:
        grads = Executor.backward('no_roots', inputs, [], [])

    assert grads is None
    assert weight.grad is None
    assert all(tensor.grad is None for tensor in inputs)
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


@pytest.mark.parametrize('use_fbw', [False, True])
def test_release_storage_keeps_backward_roots(use_fbw):
    if use_fbw and not _patch_torch.FBW_SUPPORTED:
        pytest.skip('FBW is not supported by this torch version')
    torch.manual_seed(0)
    reference, actual = _make_linears()
    input_data = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(4, 16, dtype=torch.float64)

    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad)

    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('linear', actual, actual_input)
    Executor.release_storage(output, actual.weight)
    # metadata and autograd graph survive, only the buffer is gone
    assert output.shape == (4, 16) and output.grad_fn is not None
    assert output.untyped_storage().nbytes() == output.element_size()
    # leaves are left alone
    assert actual.weight.untyped_storage().nbytes() == 16 * 8 * actual.weight.element_size()

    if use_fbw:
        input_grad = Executor.backward_input(
            'linear', [actual_input], [output], [output_grad], actual.parameters()
        )
        Executor.backward_weight('linear', actual.parameters())
    else:
        input_grad = Executor.backward('linear', [actual_input], [output], [output_grad])
    Executor.check_clear()
    torch.testing.assert_close(input_grad, reference_input.grad)
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    torch.testing.assert_close(actual.bias.grad, reference.bias.grad)
    # the grad shape check still works against the released root
    released = Executor.fexecute('linear', actual, actual_input)
    Executor.release_storage(released)
    with pytest.raises(RuntimeError, match='shape'):
        Executor.backward('linear', [actual_input], [released], [output_grad[:2]])


def test_release_storage_unwraps_zero3_backward_hook_views():
    from nnscaler.runtime.function import insert_backward_hook
    torch.manual_seed(0)
    reference, actual = _make_linears()
    input_data = torch.randn(4, 8, dtype=torch.float64)
    output_grad = torch.randn(4, 16, dtype=torch.float64)
    reference_input = input_data.clone().requires_grad_()
    reference(reference_input).backward(output_grad)

    fired = []
    def segment(x):
        # ZeRO-3 codegen wraps the output once per parameter input and runs the
        # segment under `save_params_hooks`, whose pack returns activations as is
        with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
            out = actual(x)
            out = insert_backward_hook(out, lambda: fired.append('weight'))
            return insert_backward_hook(out, lambda: fired.append('bias'))

    actual_input = input_data.clone().requires_grad_()
    output = Executor.fexecute('linear', segment, actual_input)
    assert output._base is not None
    Executor.release_storage(output)
    # both the hook view and the fresh op output behind it are released
    assert output.untyped_storage().nbytes() == output.element_size()
    assert output._base.untyped_storage().nbytes() == output.element_size()
    assert output.shape == output._base.shape == (4, 16)

    input_grad = Executor.backward('linear', [actual_input], [output], [output_grad])
    torch.testing.assert_close(input_grad, reference_input.grad)
    torch.testing.assert_close(actual.weight.grad, reference.weight.grad)
    assert fired == ['bias', 'weight']

    # a hooked view whose root base is still saved as another op's input is left alone
    base = actual_input * 2.0
    kept = base * base   # saves `base` as an input
    hooked_view = insert_backward_hook(base.reshape(base.shape), lambda: None)
    Executor.release_storage(hooked_view)
    assert base.untyped_storage().nbytes() == 4 * 8 * base.element_size()
    assert hooked_view.untyped_storage().nbytes() == 4 * 8 * base.element_size()
    del kept


def test_release_storage_skips_tensors_held_elsewhere():
    x = torch.randn(4, 3, requires_grad=True)
    nbytes = 4 * 3 * x.element_size()

    # an output that another op saved as its input (same TensorImpl) must survive;
    # e.g. a no-op `contiguous()` returns this very object as the stage output
    y = x * 2.0
    z = y * y             # saves y
    out = y.contiguous()  # same object
    assert out is y
    Executor.release_storage(out)
    assert y.untyped_storage().nbytes() == nbytes
    torch.autograd.backward([z], [torch.ones_like(z)])
    torch.testing.assert_close(x.grad, 8 * x.detach())

    # an output saved only by its own producer (shallow copy) is released and still differentiable
    x.grad = None
    t = torch.tanh(x)
    Executor.release_storage(t)
    assert t.untyped_storage().nbytes() == t.element_size()
    torch.autograd.backward([t], [torch.ones_like(t)])
    torch.testing.assert_close(x.grad, 1 - torch.tanh(x.detach()) ** 2)

    # under saved_tensors_hooks that return the tensor itself (ZeRO-3 save_params_hooks,
    # CPU offloading) the saved output is a Python reference to this very object
    x.grad = None
    with torch.autograd.graph.saved_tensors_hooks(lambda a: a, lambda a: a):
        hooked = torch.tanh(x)
    Executor.release_storage(hooked)
    assert hooked.untyped_storage().nbytes() == nbytes
    torch.autograd.backward([hooked], [torch.ones_like(hooked)])
    torch.testing.assert_close(x.grad, 1 - torch.tanh(x.detach()) ** 2)

    # a second Python name for the same object (e.g. a sibling segment output) keeps it
    first = x * 5.0
    second = first
    Executor.release_storage(first)
    assert first.untyped_storage().nbytes() == nbytes
    del second

    # a view whose base is still named elsewhere leaves both alone
    base = x * 6.0
    view = base.view(3, 4)
    Executor.release_storage(view)
    assert base.untyped_storage().nbytes() == nbytes and view.untyped_storage().nbytes() == nbytes


def test_hold_send_keeps_buffer_alive_after_release_storage():
    class _Work:
        def is_completed(self):
            return False
        def wait(self):
            pass

    handler = executor.AsyncCommHandler()
    output = torch.randn(4, 8, requires_grad=True) * 2.0
    data_ptr, nbytes = output.data_ptr(), output.untyped_storage().nbytes()
    handler.hold_send(output, _Work())
    Executor.release_storage(output)
    # the sender's tensor is emptied, but the held alias still owns the real buffer
    assert output.untyped_storage().nbytes() == output.element_size()
    held = handler._send_holds[0][0]
    assert held.data_ptr() == data_ptr and held.untyped_storage().nbytes() == nbytes
    handler.drain_sends()
    handler.check_clear()
