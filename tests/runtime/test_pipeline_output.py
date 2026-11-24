# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
import gc
import io
import weakref

import pytest
import torch
from torch.autograd.graph import saved_tensors_hooks

from nnscaler.runtime.executor import Executor
from nnscaler.runtime.module import ParallelModule
from nnscaler.runtime.pipeline_input import OutputReference


@pytest.fixture(autouse=True)
def clear_executor():
    Executor.clear()
    yield
    Executor.clear()
    gc.collect()


@pytest.mark.parametrize('release_inputs', [False, True])
@pytest.mark.parametrize('release_outputs', [False, True])
@pytest.mark.parametrize('view', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_pipeline_roots_release_payload_and_preserve_both_segment_gradients(
        release_inputs, release_outputs, view, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    x = torch.randn(8, 4, device=device, dtype=torch.float64, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn_like(x))
    def first(value):
        result = value * weight
        return result.t() if view else result
    expected_x, expected_w = torch.autograd.grad(first(x).square().sum(), (x, weight))
    # Simulate saved activation copies independently of the original storage.
    # GPU tests use real host copies here; production DMA is tested separately.
    enabled = gc.isenabled()
    gc.disable()
    try:
        with saved_tensors_hooks(lambda v: v.detach().cpu().clone(), lambda v: v.to(device)):
            middle = Executor.fexecute('first', first, x)
            storage = weakref.ref(middle.untyped_storage())
            loss = Executor.fexecute('second', lambda value: value.square().sum(), middle,
                                     release_inputs=(0,) if release_inputs else ())
        root = Executor.release_output_tensor(middle) if release_outputs else middle
        del middle
        assert (storage() is None) == (release_inputs and release_outputs)
        grad = Executor.backward('second', (root,), (loss,), (None,))
        actual_x = Executor.backward('first', (x,), (root,), (grad,))
        torch.testing.assert_close(actual_x, expected_x, rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, expected_w, rtol=0, atol=0)
        if release_outputs:
            assert root.edge is None
        Executor.check_clear()
    finally:
        if enabled:
            gc.enable()


def test_output_reference_preserves_external_aliases_and_retained_gradient():
    x = torch.randn(8, requires_grad=True)
    output = Executor.fexecute('s', lambda v: v * 2, x)
    output.retain_grad()
    alias = output.view(2, 4)
    before, pointer = output.detach().clone(), output.data_ptr()
    root = Executor.release_output_tensor(output)
    assert isinstance(root, OutputReference)
    assert output.data_ptr() == pointer and output.shape == before.shape
    torch.testing.assert_close(alias.flatten(), before)
    grad = torch.ones_like(output)
    Executor.backward('s', (x,), (root,), (grad,))
    torch.testing.assert_close(output.grad, grad, rtol=0, atol=0)
    torch.testing.assert_close(alias.flatten(), before)
    Executor.check_clear()


@pytest.mark.parametrize('distinct_view', [False, True])
def test_root_references_preserve_tensor_identity_for_duplicate_outputs(distinct_view):
    x = torch.ones(8, requires_grad=True)
    output = Executor.fexecute('s', lambda v: v * 2, x)
    other = output.view_as(output) if distinct_view else output
    roots = (Executor.release_output_tensor(output), Executor.release_output_tensor(other))
    incoming = torch.ones_like(output)
    del output, other
    actual = Executor.backward('s', (x,), roots, (incoming, incoming))
    torch.testing.assert_close(actual, torch.full_like(x, 4 if distinct_view else 2), rtol=0, atol=0)
    assert all(root.edge is None for root in roots)
    Executor.check_clear()


def test_output_reference_takes_legacy_send_ownership_without_mutating_the_tensor():
    x = torch.ones(8, requires_grad=True)
    output = Executor.fexecute('s', lambda v: v * 2, x)
    pointer = output.data_ptr()
    Executor.defer_pseudo_free_tensor(output)
    Executor.defer_pseudo_free_tensor(output)
    assert len(Executor._pseudo_free_grad_edges) == 1
    root = Executor.release_output_tensor(output)
    assert not Executor._pseudo_free_grad_edges and not Executor._pseudo_free_pending_sends
    # Async send owners may complete after the schedule has captured the edge.
    Executor.complete_deferred_pseudo_free_tensor(output)
    Executor.complete_deferred_pseudo_free_tensor(output)
    assert output.shape == (8,) and output.data_ptr() == pointer
    torch.testing.assert_close(output, torch.full_like(output, 2))
    actual = Executor.backward('s', (x,), (root,), (torch.ones_like(output),))
    torch.testing.assert_close(actual, torch.full_like(x, 2), rtol=0, atol=0)
    Executor.check_clear()


def test_dead_output_reference_does_not_collide_with_a_reused_tensor_id():
    x = torch.ones(8, requires_grad=True)
    output = Executor.fexecute('s', lambda v: v * 2, x)
    root = Executor.release_output_tensor(output)
    del output
    other = torch.ones(8, requires_grad=True)
    root.tensor_id = id(other)  # Deterministically simulate Python id reuse.
    incoming = torch.ones(8)
    actual = Executor.backward('s', (x,), (root, other), (incoming, incoming))
    torch.testing.assert_close(actual, torch.full_like(x, 2), rtol=0, atol=0)
    torch.testing.assert_close(other.grad, incoming, rtol=0, atol=0)
    Executor.check_clear()


def test_reference_identity_is_weak_and_does_not_change_tensor_serialization():
    from nnscaler.runtime.pipeline_input import InputReference
    tensor = torch.ones(8, requires_grad=True)
    attributes = dict(vars(tensor))
    a, b = InputReference(tensor), InputReference(tensor)
    assert a.identity is b.identity
    observed = weakref.ref(a.identity)
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    torch.testing.assert_close(torch.load(buffer, weights_only=True), tensor)
    assert vars(tensor) == attributes
    del a, b
    assert observed() is None


def test_output_saved_by_autograd_stays_alive_until_consumed():
    x = torch.randn(8, requires_grad=True)
    output = Executor.fexecute('s', torch.sigmoid, x)
    expected = output.detach().clone() * (1 - output.detach().clone())
    storage = weakref.ref(output.untyped_storage())
    root = Executor.release_output_tensor(output)
    del output
    assert storage() is not None
    actual = Executor.backward('s', (x,), (root,), (torch.ones(8),))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert storage() is None
    Executor.check_clear()


@pytest.mark.parametrize('reason', ['leaf', 'scalar', 'api', 'hook'])
def test_output_release_falls_back_when_data_or_api_is_required(monkeypatch, reason):
    x = torch.ones(8, requires_grad=True)
    output = x * 2
    if reason == 'leaf':
        output = x
    elif reason == 'scalar':
        output = output.sum()
    elif reason == 'api':
        monkeypatch.setattr('nnscaler.runtime.executor.get_gradient_edge', None)
    else:
        Executor.register_backward_pre_hook(lambda ins, outs, grads: (ins, outs, grads))
    assert Executor.release_output_tensor(output) is output


def test_changing_to_a_data_reading_hook_after_release_is_rejected():
    x = torch.randn(8, requires_grad=True)
    output = Executor.fexecute('s', lambda v: v * 2, x)
    root = Executor.release_output_tensor(output)
    Executor.register_backward_pre_hook(lambda ins, outs, grads: (ins, outs, grads))
    with pytest.raises(RuntimeError, match='must support output references'):
        Executor.backward('s', (x,), (root,), (torch.ones_like(output),))


def test_loss_scaling_keeps_scalar_losses_and_accepts_released_intermediate_roots():
    ParallelModule._scale_loss(object(), None, lambda loss: loss * 3)
    x = torch.randn(8, requires_grad=True)
    middle = Executor.fexecute('first', lambda v: v * 2, x)
    loss = Executor.fexecute('second', torch.sum, middle, release_inputs=(0,))
    root = Executor.release_output_tensor(middle)
    assert isinstance(root, OutputReference)
    assert Executor.release_output_tensor(loss) is loss
    grad = Executor.backward('second', (root,), (loss,), (None,))
    actual = Executor.backward('first', (x,), (root,), (grad,))
    torch.testing.assert_close(actual, torch.full_like(x, 6), rtol=0, atol=0)
    Executor.check_clear()
