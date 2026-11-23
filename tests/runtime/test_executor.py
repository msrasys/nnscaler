#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging
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
        def wait(self):
            events.append('wait')
            pending.copy_(torch.arange(8, dtype=torch.float32))

    handler.submit(pending, [Work()])

    def adapter(tensor):
        events.append('consume')
        return tensor * 2

    result = Executor.aexecute(adapter, pending, requires_grad=requires_grad)
    assert events == ['wait', 'consume']
    torch.testing.assert_close(result, torch.arange(8, dtype=torch.float32) * 2)
    assert result.requires_grad == requires_grad
    handler.check_clear()


def _submit_gather_like(handler):
    """Mimic all_gather: the handle is the preallocated output, filled by the callback."""
    output = torch.empty(8)
    gathered = [torch.empty(4), torch.empty(4)]

    class Work:
        def wait(self):
            for part in gathered:
                part.copy_(torch.arange(4, dtype=torch.float32))

    def assemble(out):
        with torch.no_grad():
            torch.cat(gathered, out=out)
        return out

    def adapter():
        handler.submit(output, [Work()], assemble)
        return output

    return Executor.aexecute(adapter, requires_grad=True)


def test_async_output_consumed_twice_in_one_call(monkeypatch):
    handler = executor._AsyncCommHandler()
    monkeypatch.setattr(executor, '_instance', handler)
    output = _submit_gather_like(handler)
    assert output.requires_grad
    first, second = Executor.sync_tensors([output, output])
    assert first is output and second is output
    torch.testing.assert_close(output.detach(), torch.arange(4, dtype=torch.float32).repeat(2))
    handler.check_clear()


def test_async_output_shared_across_calls(monkeypatch, caplog):
    handler = executor._AsyncCommHandler()
    monkeypatch.setattr(executor, '_instance', handler)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    output = _submit_gather_like(handler)
    # An adapter consumes the output first; a later segment and its backward
    # must still see the assembled tensor, not a stale placeholder.
    assert Executor.aexecute(lambda t: t, output, requires_grad=False) is output
    loss = Executor.fexecute('seg', lambda x: (x * 2).sum(), output, requires_grad=True)
    with caplog.at_level(logging.WARNING, logger='nnscaler.runtime.executor'):
        grad = Executor.backward('seg', [output], [loss], [torch.ones(())])
    assert "doesn't match" not in caplog.text
    torch.testing.assert_close(loss.detach(), torch.tensor(24.0))
    torch.testing.assert_close(grad, torch.full((8,), 2.0))
    handler.check_clear()
    Executor.check_clear()


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
    # A segment without gradient roots reports "no gradient" to registered callbacks.
    callback_grads = []
    for tensor in inputs:
        Executor.register_input_grad_callback(tensor, callback_grads.append)
    if use_fbw:
        grads = Executor.backward_input('no_roots', inputs, [], [], [weight])
        assert len(Executor._weight_backward_states['no_roots']) == 1
        Executor.backward_weight('no_roots', [weight])
    else:
        grads = Executor.backward('no_roots', inputs, [], [])

    assert grads is None
    assert weight.grad is None
    assert all(tensor.grad is None for tensor in inputs)
    assert callback_grads == [None] * ninputs
    Executor.check_clear()


@pytest.mark.parametrize('mode', ['full', 'split', 'custom_fbw'])
def test_input_grad_callback_accumulates_consumer_segments(mode):
    weight = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
    outer_input = weight * 3
    outer_input.retain_grad()
    callback_grads = []

    def resume_outer_backward(grad):
        callback_grads.append(grad)
        torch.autograd.backward(outer_input, grad)

    Executor.register_input_grad_callback(outer_input, resume_outer_backward)
    first = Executor.fexecute('first', lambda tensor: tensor * 5, outer_input)
    # A consumer without gradient roots adds nothing to the others' sum.
    Executor.fexecute('rootless', lambda tensor: (tensor * 11).data, outer_input)
    second = Executor.fexecute('second', lambda tensor: tensor * 7, outer_input)
    # custom_fbw wraps a replacement that dispatches through Executor itself;
    # the pending entry must not be accumulated a second time by the wrapper.
    for name, outputs in (('second', [second]), ('rootless', []), ('first', [first])):
        output_grads = [torch.ones_like(output) for output in outputs]
        if mode == 'full':
            Executor.backward(name, [outer_input], outputs, output_grads)
        else:
            with executor.custom_fbw(Executor.backward_input, Executor.backward_weight) \
                    if mode == 'custom_fbw' else torch.enable_grad():
                executor.backward_input(name, [outer_input], outputs, output_grads, [])
                executor.backward_weight(name, [])
        # Only the last pending consumer fires the callback.
        assert len(callback_grads) == (name == 'first')

    torch.testing.assert_close(callback_grads[0], torch.tensor(12., dtype=torch.float64))
    torch.testing.assert_close(weight.grad, torch.tensor(36., dtype=torch.float64))
    # The callback's outer backward retains the grad; Executor must not add a second copy.
    torch.testing.assert_close(outer_input.grad, torch.tensor(12., dtype=torch.float64))
    Executor.check_clear()


@pytest.mark.parametrize('bypass_executor', [False, True])
@pytest.mark.parametrize('callback', [False, True])
@pytest.mark.parametrize('check_coverage', [False, True])
def test_custom_fbw_handles_boundary_grads(check_coverage, callback, bypass_executor):
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

    # A replacement that consumes _detach itself needs the wrapper to write the
    # retained grad, or dispatch the callback, which then writes it once.
    # One that calls Executor must not get them a second time from the wrapper.
    with executor.custom_fbw(
        custom_input if bypass_executor else Executor.backward_input,
        custom_weight if bypass_executor else Executor.backward_weight,
        check_grad_coverage=check_coverage,
    ):
        # Registration happens inside the context, as in a generated schedule.
        if callback:
            Executor.register_input_grad_callback(boundary, resume)
        grad = executor.backward_input(
            'segment', [boundary], [output], [torch.ones_like(output)], [weight],
        )
        assert len(callbacks) == callback
        assert weight.grad is None
        executor.backward_weight('segment', [weight])
    torch.testing.assert_close(grad, reference_input.grad)
    torch.testing.assert_close(boundary.grad, reference_input.grad)
    if callback:
        torch.testing.assert_close(leaf.grad, reference_input.grad * 2)
    else:
        assert leaf.grad is None
    torch.testing.assert_close(weight.grad, reference_weight.grad)
    Executor.check_clear()


def test_input_grad_callback_rejects_consumer_after_delivery():
    shared = torch.randn(3, requires_grad=True) * 1
    callback_grads = []

    def run_micro_batch():
        output = Executor.fexecute('segment', lambda tensor: tensor * 5, shared)
        Executor.backward('segment', [shared], [output], [torch.ones_like(output)])

    Executor.register_input_grad_callback(shared, callback_grads.append)
    run_micro_batch()
    assert len(callback_grads) == 1
    # Feeding the same tensor to a later micro-batch would lose that gradient.
    with pytest.raises(RuntimeError, match='callback already fired'):
        Executor.fexecute('segment', lambda tensor: tensor * 5, shared)

    # Registering again starts a new round.
    Executor.register_input_grad_callback(shared, callback_grads.append)
    run_micro_batch()
    assert len(callback_grads) == 2

    # A released tensor is forgotten, so a new tensor reusing its id is not rejected.
    del shared
    assert not Executor._delivered_input_grads
    Executor.check_clear()


def test_custom_fbw_survives_reentrant_backward_in_callback():
    first = torch.tensor([3.0], requires_grad=True) * 1
    first.retain_grad()
    second = torch.tensor([4.0], requires_grad=True) * 1
    first_output = Executor.fexecute('first', lambda tensor: tensor * 5, first)
    second_output = Executor.fexecute('second', lambda tensor: tensor * 7, second)

    def run_second_segment(grad):
        # The callback runs another segment's backward through Executor.
        Executor.backward('second', [second], [second_output], [torch.ones(1)])

    with executor.custom_fbw(Executor.backward_input, Executor.backward_weight):
        Executor.register_input_grad_callback(first, run_second_segment)
        executor.backward_input('first', [first], [first_output], [torch.ones(1)], [])
        executor.backward_weight('first', [])
    # The wrapper must still see the first segment as handled by Executor,
    # rather than write the retained grad the callback owns.
    assert first.grad is None
    Executor.check_clear()


@pytest.mark.parametrize('split', [False, True])
def test_input_grad_callback_accumulates_before_reentrant_backward(split):
    first = torch.tensor([1.0], requires_grad=True) * 1
    shared = torch.tensor([1.0], requires_grad=True) * 1
    first_output = Executor.fexecute('first', lambda x, y: x * 2 + y * 3, first, shared)
    second_output = Executor.fexecute('second', lambda y: y * 5, shared)
    shared_grads = []

    def run_second_segment(grad):
        # Runs the other consumer of ``shared`` before the first segment
        # has dispatched ``shared`` itself.
        Executor.backward('second', [shared], [second_output], [torch.ones(1)])

    Executor.register_input_grad_callback(first, run_second_segment)
    Executor.register_input_grad_callback(shared, shared_grads.append)
    if split:
        Executor.backward_input('first', [first, shared], [first_output], [torch.ones(1)], [])
        Executor.backward_weight('first', [])
    else:
        Executor.backward('first', [first, shared], [first_output], [torch.ones(1)])
    # Both segments contribute before the shared callback fires, exactly once.
    assert [grad.item() for grad in shared_grads] == [3.0 + 5.0]
    Executor.check_clear()


def test_check_input_grad_callbacks_fired_rejects_pending_callback():
    # E.g. registered on a rank whose pipeline stage never consumes the tensor.
    tensor = torch.randn(2, requires_grad=True) * 1
    Executor.register_input_grad_callback(tensor, lambda grad: None)
    with pytest.raises(RuntimeError, match='did not fire'):
        Executor.check_input_grad_callbacks_fired()
    # The pending callback is dropped so that the outer graph is released.
    Executor.check_clear()


def test_custom_fbw_rejects_missing_input_grad_for_callback():
    boundary = torch.randn(3, requires_grad=True) * 1
    output = Executor.fexecute('segment', lambda tensor: tensor * 5, boundary)
    Executor.register_input_grad_callback(boundary, lambda grad: None)

    def custom_input(name, inputs, outputs, grads, weights):
        # Drops the input gradient although the segment has gradient roots.
        Executor._detach[name].pop(0)
        return None

    with executor.custom_fbw(custom_input, lambda name, weights: None):
        with pytest.raises(RuntimeError, match='returned no gradient for a retained or callback input'):
            executor.backward_input('segment', [boundary], [output], [torch.ones_like(output)], [])


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


def _segment_backward(use_fbw, name, inputs, outputs, grads):
    if not use_fbw:
        return Executor.backward(name, inputs, outputs, grads)
    result = Executor.backward_input(name, inputs, outputs, grads, [])
    Executor.backward_weight(name, [])
    return result


@pytest.mark.parametrize('use_fbw', [False, True])
@pytest.mark.parametrize('repeats', [1, 2, 3])
def test_backward_retains_repeated_boundary_input_once(repeats, use_fbw):
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
        grads = _segment_backward(
            use_fbw, 'segment', [outer_input] * repeats, [output], [torch.ones_like(output)],
        )
        for grad in (grads,) if repeats == 1 else grads:
            torch.testing.assert_close(grad, expected)
        torch.testing.assert_close(outer_input.grad, reference.grad)
        assert leaf.grad is None
    Executor.check_clear()


@pytest.mark.parametrize('use_fbw', [False, True])
def test_backward_preserves_retained_outer_input_grad(use_fbw):
    Executor.clear()
    try:
        leaf = torch.randn(4, 3, requires_grad=True)
        outer_input = leaf * 2
        outer_input.retain_grad()

        output = Executor.fexecute('segment', torch.sin, outer_input)
        expected = torch.cos(outer_input)
        actual = _segment_backward(
            use_fbw, 'segment', (outer_input,), (output,), (torch.ones_like(output),),
        )

        assert torch.equal(actual, expected)
        assert torch.equal(outer_input.grad, expected)
        assert leaf.grad is None
        Executor.check_clear()
    finally:
        Executor.clear()
