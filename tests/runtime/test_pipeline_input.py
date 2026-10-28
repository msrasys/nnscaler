# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
import gc
import weakref

import pytest
import torch
from torch.autograd.graph import saved_tensors_hooks

from nnscaler.runtime.executor import Executor, release_input_tensor
from nnscaler.runtime.pipeline_input import InputReference, InputGradSlot


@pytest.fixture(autouse=True)
def clear_executor():
    Executor.clear()
    yield
    Executor.clear()
    gc.collect()


@pytest.mark.parametrize('release', [False, True])
def test_input_storage_lifetime_and_parameter_gradients(release):
    torch.manual_seed(42)
    x = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(4, dtype=torch.float64))
    def compute(value):
        return (value.sin() * weight).cos().sum()
    dx, dw = torch.autograd.grad(compute(x), (x, weight))
    storage = weakref.ref(x.untyped_storage())
    enabled = gc.isenabled()
    gc.disable()
    try:
        # Independent saved copies simulate the host copy without a CUDA
        # context. The real GPU test separately checks allocator reclamation.
        with saved_tensors_hooks(lambda v: v.detach().clone(), lambda v: v):
            output = Executor.fexecute('s', compute, x, release_inputs=(0,) if release else ())
        targets = Executor._detach['s'][0]
        slot = targets[0][1]
        reference = release_input_tensor(x) if release else x
        del x
        if release:
            assert storage() is None
            assert isinstance(slot, InputGradSlot)
        else:
            assert storage() is not None
        actual = Executor.backward('s', (reference,), (output,), (None,))
        torch.testing.assert_close(actual, dx, rtol=0, atol=0)
        torch.testing.assert_close(weight.grad, dw, rtol=0, atol=0)
        if release:
            assert slot.grad is None  # An externally kept output owns no dInput.
        Executor.check_clear()
    finally:
        if enabled:
            gc.enable()


def test_duplicate_inputs_share_one_gradient_slot():
    x = torch.randn(4, 3, requires_grad=True)
    expected = torch.autograd.grad(((x + x).sin()).sum(), x)[0]
    y = Executor.fexecute('s', lambda a, b: (a + b).sin().sum(), x, x, release_inputs=(0, 1))
    pairs = Executor._detach['s'][0]
    assert pairs[0][1] is pairs[1][1]
    ref = release_input_tensor(x)
    left, right = Executor.backward('s', (ref, ref), (y,), (None,))
    assert left is right
    torch.testing.assert_close(left, expected, rtol=0, atol=0)
    Executor.check_clear()


@pytest.mark.parametrize('reentrant', [False, True])
def test_checkpoint_replay_preserves_input_and_parameter_gradients(reentrant):
    from torch.utils.checkpoint import checkpoint
    x = torch.randn(4, 3, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(3))
    def compute(value):
        return (value.sin() * weight).cos().sum()
    dx, dw = torch.autograd.grad(compute(x), (x, weight))
    output = Executor.fexecute('s', lambda value: checkpoint(compute, value, use_reentrant=reentrant),
                               x, release_inputs=(0,))
    actual = Executor.backward('s', (release_input_tensor(x),), (output,), (None,))
    torch.testing.assert_close(actual, dx, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, dw, rtol=0, atol=0)
    Executor.check_clear()


def test_alias_and_noncontiguous_inputs_keep_independent_gradients():
    x = torch.randn(4, 4)
    a, b = x[:, :2].requires_grad_(), x[:, 1:3].requires_grad_()
    ra, rb = a.detach().requires_grad_(), b.detach().requires_grad_()
    expected = torch.autograd.grad((ra.sin() * rb.cos()).sum(), (ra, rb))
    y = Executor.fexecute('s', lambda left, right: (left.sin() * right.cos()).sum(), a, b,
                          release_inputs=(0, 1))
    result = Executor.backward('s', (release_input_tensor(a), release_input_tensor(b)), (y,), (None,))
    for actual, reference in zip(result, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    Executor.check_clear()


@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'slice'])
def test_input_gradients_keep_leaf_layout_and_do_not_alias_incoming_gradient(layout):
    source = torch.randn(4, 4)
    if layout == 'transpose':
        source = source.t()
    elif layout == 'slice':
        source = source[:, :2]
    x = source.requires_grad_()
    gradients = []
    for release in (False, True):
        incoming = torch.tensor(2.0)
        y = Executor.fexecute('s', lambda value: value.sum(), x, release_inputs=(0,) if release else ())
        grad = Executor.backward('s', (release_input_tensor(x) if release else x,), (y,), (incoming,))
        gradients.append(grad)
        assert grad.untyped_storage()._cdata != incoming.untyped_storage()._cdata
        grad.add_(1)  # A caller can reuse its input gradient independently.
        assert incoming.item() == 2
    assert gradients[0].stride() == gradients[1].stride()
    torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
    Executor.check_clear()


def test_retained_outer_input_and_callback_are_preserved():
    weight = torch.nn.Parameter(torch.tensor(2.0))
    outer = weight * 3
    outer.retain_grad()
    y = Executor.fexecute('s', torch.sin, outer, release_inputs=(0,))
    grads = []
    def callback(grad):
        grads.append(grad)
        torch.autograd.backward(outer, grad)
    Executor.register_input_grad_callback(outer, callback)
    actual = Executor.backward('s', (release_input_tensor(outer),), (y,), (torch.ones_like(y),))
    expected = torch.cos(outer.detach())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Both the explicit retain_grad copy and the nested outer backward add it,
    # matching the existing executor semantics when callers combine the two.
    torch.testing.assert_close(outer.grad, 2 * expected, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, 3 * expected, rtol=0, atol=0)
    assert len(grads) == 1
    Executor.check_clear()


def test_id_reuse_does_not_consume_a_new_tensors_callback():
    x = torch.randn(4, requires_grad=True)
    y = Executor.fexecute('old', lambda value: (value * 2).sum(), x, release_inputs=(0,))
    key = Executor._detach['old'][0][0][0]
    ref = release_input_tensor(x)
    del x
    assert key.original() is None
    new = torch.randn(4, requires_grad=True)
    calls = []
    Executor.register_input_grad_callback(new, calls.append)
    # Deterministically simulate allocator id reuse; no probabilistic loop.
    key.tensor_id = ref.tensor_id = id(new)
    Executor.backward('old', (ref,), (y,), (None,))
    assert calls == [] and id(new) in Executor._input_grad_callbacks
    z = Executor.fexecute('new', lambda value: (value * 3).sum(), new, release_inputs=(0,))
    Executor.backward('new', (release_input_tensor(new),), (z,), (None,))
    assert len(calls) == 1 and torch.equal(calls[0], torch.full_like(new, 3))
    Executor.check_clear()


def test_weak_outer_graph_callback_releases_data_before_backward():
    weight = torch.nn.Parameter(torch.randn(4))
    feature = weight * 3
    edge = torch.autograd.graph.get_gradient_edge(feature)
    storage = weakref.ref(feature.untyped_storage())
    owners = {'feature': feature}
    calls = []
    def backward(grad):
        nonlocal edge
        calls.append('backward')
        torch.autograd.backward(edge, grad)
        edge = None
    def release():
        calls.append('release')
        owners.clear()
    Executor.register_input_grad_callback(feature, backward, release_input=True, on_data_release=release)
    output = Executor.fexecute('s', torch.sum, feature, release_inputs=(0,))
    reference = release_input_tensor(feature)
    del feature
    assert storage() is None and not owners and calls == ['release']
    assert release_input_tensor(reference) is reference
    grad = Executor.backward('s', (reference,), (output,), (None,))
    assert calls == ['release', 'backward']
    torch.testing.assert_close(grad, torch.ones(4), rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, torch.full((4,), 3.0), rtol=0, atol=0)
    Executor.check_clear()


def test_weak_callbacks_with_reused_tensor_ids_remain_distinct():
    old = torch.randn(4, requires_grad=True)
    calls = []
    Executor.register_input_grad_callback(old, lambda grad: calls.append(('old', grad.clone())), release_input=True)
    output = Executor.fexecute('old', torch.sum, old, release_inputs=(0,))
    key = Executor._detach['old'][0][0][0]
    reference = release_input_tensor(old)
    del old
    assert key.original() is None
    new = torch.randn(4, requires_grad=True)
    key.tensor_id = reference.tensor_id = id(new)
    Executor.register_input_grad_callback(new, lambda grad: calls.append(('new', grad.clone())), release_input=True)
    later = Executor.fexecute('new', lambda value: (value * 2).sum(), new, release_inputs=(0,))
    Executor.backward('new', (release_input_tensor(new),), (later,), (None,))
    Executor.backward('old', (reference,), (output,), (None,))
    assert [name for name, _ in calls] == ['new', 'old']
    torch.testing.assert_close(calls[0][1], torch.full((4,), 2.0), rtol=0, atol=0)
    torch.testing.assert_close(calls[1][1], torch.ones(4), rtol=0, atol=0)
    Executor.check_clear()


def test_weak_callback_runs_on_first_backward_consumer_once():
    x = torch.randn(4, requires_grad=True)
    calls = []
    Executor.register_input_grad_callback(x, calls.append, release_input=True)
    first = Executor.fexecute('first', torch.sum, x, release_inputs=(0,))
    second = Executor.fexecute('second', lambda value: (value * 2).sum(), x, release_inputs=(0,))
    reference = release_input_tensor(x)
    Executor.backward('second', (reference,), (second,), (None,))
    Executor.backward('first', (reference,), (first,), (None,))
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], torch.full((4,), 2.0), rtol=0, atol=0)
    Executor.check_clear()


def test_weak_callback_requires_registration_before_forward():
    x = torch.randn(4, requires_grad=True)
    y = Executor.fexecute('s', torch.sin, x, release_inputs=(0,))
    with pytest.raises(RuntimeError, match='before forward'):
        Executor.register_input_grad_callback(x, lambda grad: None, release_input=True)
    Executor.backward('s', (release_input_tensor(x),), (y,), (torch.ones_like(y),))
    Executor.check_clear()


def test_unknown_backward_hook_keeps_tensor_targets():
    def hook(inputs, outputs, grads):
        assert all(torch.is_tensor(value) and value.is_leaf for value in inputs)
        return inputs, outputs, grads
    Executor.register_backward_pre_hook(hook)
    x = torch.randn(4, requires_grad=True)
    y = Executor.fexecute('s', torch.sin, x, release_inputs=(0,))
    assert torch.is_tensor(Executor._detach['s'][0][0][1])
    actual = Executor.backward('s', (release_input_tensor(x),), (y,), (torch.ones_like(y),))
    torch.testing.assert_close(actual, torch.cos(x), rtol=0, atol=0)
    Executor.check_clear()


def test_declared_output_only_hook_scales_slot_and_weight_gradients():
    def hook(inputs, outputs, grads):
        assert isinstance(inputs[0], InputGradSlot)
        return inputs, [outputs[0] * 3], grads
    hook._nnscaler_supports_input_grad_slots = True
    Executor.register_backward_pre_hook(hook)
    x = torch.randn(4, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(4))
    y = Executor.fexecute('s', lambda value: (value * weight).sum(), x, release_inputs=(0,))
    actual = Executor.backward('s', (release_input_tensor(x),), (y,), (None,))
    torch.testing.assert_close(actual, weight * 3, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, x * 3, rtol=0, atol=0)
    Executor.check_clear()


def test_hook_changed_after_forward_is_rejected():
    x = torch.randn(4, requires_grad=True)
    y = Executor.fexecute('s', torch.sin, x, release_inputs=(0,))
    Executor.register_backward_pre_hook(lambda ins, outs, grads: (ins, outs, grads))
    with pytest.raises(RuntimeError, match='hook changed'):
        Executor.backward('s', (release_input_tensor(x),), (y,), (torch.ones_like(y),))


def test_split_backward_with_released_inputs_is_rejected():
    x = torch.randn(4, requires_grad=True)
    y = Executor.fexecute('s', torch.sin, x, release_inputs=(0,))
    with pytest.raises(RuntimeError, match='do not support split backward'):
        Executor.backward_input('s', (release_input_tensor(x),), (y,), (torch.ones_like(y),), ())


def test_compile_config_rejects_input_release_with_split_backward():
    from nnscaler.parallel import ComputeConfig
    with pytest.raises(ValueError, match='not supported with split backward'):
        ComputeConfig(2, 2, use_end2end=True, use_fbw=True,
                      cpu_offload={'release_pipeline_inputs': True})


def test_forward_failure_discards_only_its_own_pending_input_state():
    x = torch.randn(4, requires_grad=True)
    first = Executor.fexecute('s', torch.sin, x, release_inputs=(0,))
    def broken(value):
        raise RuntimeError('forward failure')
    with pytest.raises(RuntimeError, match='forward failure'):
        Executor.fexecute('s', broken, x, release_inputs=(0,))
    assert len(Executor._detach['s']) == 1
    Executor.backward('s', (release_input_tensor(x),), (first,), (torch.ones_like(first),))
    Executor.check_clear()


@pytest.mark.parametrize('indices', [(True,), (-1,), (1,), ('0',)])
def test_invalid_released_argument_indices(indices):
    with pytest.raises(ValueError, match='argument indices'):
        Executor.fexecute('s', torch.sin, torch.ones(2), release_inputs=indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_real_offload_and_input_references_release_cuda_payload():
    from types import SimpleNamespace
    from nnscaler.runtime.cpu_offloading import CPUOffloadConfig, _pools
    from nnscaler.runtime.module import ParallelModule, _ModuleTensorRegistry
    _pools.clear()
    class Model(ParallelModule, skip_init=True):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(torch.randn(2048, device='cuda'))
            self.compute_config = SimpleNamespace(cpu_offload=CPUOffloadConfig(
                max_cpu_bytes=128 * 2**20, min_tensor_bytes=0, host_reserve_bytes=0))
            self._module_tensor_registry = _ModuleTensorRegistry(self)
    model = Model()
    def compute(value):
        return (value.sin() * model.weight).cos().sum()
    def forward(value):
        with model.cpu_offloading_hooks() as context:
            result = compute(value)
        context.watch(result)
        return result
    usage = []
    for release in (False, True):
        model.weight.grad = None
        x = torch.randn(2048, 2048, device='cuda', requires_grad=True)
        dx, dw = torch.autograd.grad(compute(x), (x, model.weight))
        expected_dx, expected_dw = dx.cpu(), dw.cpu()
        del dx, dw
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated() - x.numel() * x.element_size()
        with model.activation_offload_scope(release):
            output = Executor.fexecute('s', forward, x, release_inputs=(0,) if release else ())
        reference = release_input_tensor(x) if release else x
        del x
        torch.cuda.synchronize()
        usage.append(torch.cuda.memory_allocated() - base)
        actual = Executor.backward('s', (reference,), (output,), (None,))
        model.finish_activation_offload(release)
        torch.testing.assert_close(actual.cpu(), expected_dx, rtol=0, atol=0)
        torch.testing.assert_close(model.weight.grad.cpu(), expected_dw, rtol=0, atol=0)
        assert model.activation_offload_stats()['live_cpu_bytes'] == 0
        del reference, output, actual
        torch.cuda.synchronize()
        Executor.check_clear()
    assert usage[0] - usage[1] >= 16 * 2**20, usage


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.cuda is None, reason='NVIDIA CUDA required')
def test_released_unsaved_input_records_its_forward_consumer_stream():
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    shape = (1024, 1024)
    with torch.cuda.stream(producer):
        # Make enough cached allocations to exercise reuse without relying
        # on a cold cudaMalloc's possible implicit device synchronization.
        buffers = [torch.ones(shape, device='cuda') for _ in range(9)]
        buffers[0].sum()
        x = buffers.pop(0).requires_grad_()
    producer.synchronize()
    del buffers
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(200_000_000)
        output = Executor.fexecute('s', lambda value: value.sum(), x, release_inputs=(0,))
    reference = release_input_tensor(x)
    del x
    with torch.cuda.stream(producer):
        pressure = [torch.empty(shape, device='cuda') for _ in range(9)]
        for value in pressure:
            value.fill_(9)
    torch.cuda.synchronize()
    assert output.item() == shape[0] * shape[1]
    actual = Executor.backward('s', (reference,), (output,), (None,))
    torch.testing.assert_close(actual, torch.ones(shape, device='cuda'), rtol=0, atol=0)
    Executor.check_clear()
