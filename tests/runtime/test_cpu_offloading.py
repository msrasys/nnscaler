import time

import pytest
import torch

from nnscaler.runtime import device
from nnscaler.runtime.cpu_offloading import (
    CPUOffloadContext,
    _get_prefetch_stream,
    _ModuleTensorRef,
    _OffloadBatch,
    _OffloadedTensor,
    _PREFETCH_STREAM_NAME,
)
from nnscaler.runtime.module import _ModuleTensorRegistry, ParallelModule


class _TestParallelModule(ParallelModule, skip_init=True):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self._module_tensor_registry = None


class _ConfiguredTestParallelModule(ParallelModule, skip_init=True):
    def __init__(self):
        ParallelModule.__init__(self)


def _cpu_offload_context(
    module: torch.nn.Module,
    prefetch_level: int = 2,
) -> CPUOffloadContext:
    if getattr(module, '_module_tensor_registry', None) is None:
        module._module_tensor_registry = _ModuleTensorRegistry(module)
    return CPUOffloadContext(module, prefetch_level=prefetch_level)


def _offloaded_handles(context: CPUOffloadContext) -> list[_OffloadedTensor]:
    return context.batch.handles


def test_cpu_offload_prefetch_level_configuration(monkeypatch):
    monkeypatch.delenv(ParallelModule._PREFETCH_LEVEL_ENV_VAR, raising=False)
    module = _ConfiguredTestParallelModule()
    assert module.cpu_offloading_prefetch_level == ParallelModule._PREFETCH_LEVEL_DEFAULT
    assert module.cpu_offloading_hooks().prefetch_level == ParallelModule._PREFETCH_LEVEL_DEFAULT

    monkeypatch.setenv(ParallelModule._PREFETCH_LEVEL_ENV_VAR, '3')
    assert module.cpu_offloading_prefetch_level == ParallelModule._PREFETCH_LEVEL_DEFAULT

    configured_module = _ConfiguredTestParallelModule()
    assert configured_module.cpu_offloading_prefetch_level == 3
    configured_module.cpu_offloading_prefetch_level = 1
    assert configured_module.cpu_offloading_hooks().prefetch_level == 1


@pytest.mark.parametrize('value', ['invalid', '-1'])
def test_cpu_offload_rejects_invalid_prefetch_level_env(monkeypatch, value):
    monkeypatch.setenv(ParallelModule._PREFETCH_LEVEL_ENV_VAR, value)
    with pytest.raises(ValueError, match=ParallelModule._PREFETCH_LEVEL_ENV_VAR):
        _ConfiguredTestParallelModule()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_cpu_offload_uses_runtime_prefetch_stream():
    assert _get_prefetch_stream() is device.DeviceGroup().get_stream(_PREFETCH_STREAM_NAME)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason='lack of multiple gpu devices',
)
def test_cpu_offload_rejects_multiple_devices():
    context = _cpu_offload_context(_TestParallelModule())
    with context:
        expected_device = context.device
        assert expected_device is not None
        assert expected_device.index is not None
        context._pack(torch.randn(8, device=expected_device))

        other_device = torch.device(
            'cuda', (expected_device.index + 1) % torch.cuda.device_count()
        )
        with pytest.raises(ValueError, match='only supports one CUDA device'):
            context._pack(torch.randn(8, device=other_device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_failed_cpu_offload_context_does_not_link_batch():
    previous_last_batch = _OffloadBatch.last_batch
    _OffloadBatch.last_batch = None
    try:
        context = _cpu_offload_context(_TestParallelModule())
        with pytest.raises(RuntimeError, match='forward failed'):
            with context:
                context._pack(torch.randn(8, device='cuda'))
                raise RuntimeError('forward failed')

        assert _OffloadBatch.last_batch is None
    finally:
        _OffloadBatch.last_batch = previous_last_batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_cpu_offload_skips_module_state_and_views():
    module = _TestParallelModule()
    module.register_parameter('parameter', torch.nn.Parameter(torch.randn(8, 8, device='cuda')))
    module.register_buffer('buffer', torch.randn(8, 8, device='cuda'))
    parameter = module.parameter
    buffer = module.buffer
    activation = torch.randn(8, 8, device='cuda', requires_grad=True)

    assert module._module_tensor_registry is None
    context = _cpu_offload_context(module)
    next_context = _cpu_offload_context(module)
    assert context.module is next_context.module is module
    assert module._module_tensor_registry is not None
    assert '_module_tensor_registry' not in module._parameters
    assert '_module_tensor_registry' not in module._buffers
    assert '_module_tensor_registry' not in module._modules
    with context:
        packed_parameter = context._pack(parameter)
        packed_parameter_view = context._pack(parameter.T)
        packed_buffer_view = context._pack(buffer[:, 1:])
        packed_activation = context._pack(activation)

    assert isinstance(packed_parameter, _ModuleTensorRef)
    assert isinstance(packed_parameter_view, _ModuleTensorRef)
    assert isinstance(packed_buffer_view, _ModuleTensorRef)
    assert isinstance(packed_activation, _OffloadedTensor)
    assert len(_offloaded_handles(context)) == 1
    assert packed_parameter_view.unpack().stride() == parameter.T.stride()
    assert torch.equal(packed_buffer_view.unpack(), buffer[:, 1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_cpu_offload_detects_inplace_module_tensor_mutation():
    module = _TestParallelModule()
    module.register_buffer('scale', torch.tensor([2.0, 3.0], device='cuda'))
    tensor = torch.ones(2, device='cuda', requires_grad=True)

    with _cpu_offload_context(module):
        output = tensor * module.scale

    module.scale.add_(10)

    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        output.sum().backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_cpu_offload_saves_stale_parameter_views_as_metadata():
    module = _TestParallelModule()
    parameter = torch.nn.Parameter(torch.randn(8, 8, device='cuda'))
    module.register_parameter('parameter', parameter)
    stale_view = parameter.T
    full_parameter = parameter.detach().clone()

    parameter.data = torch.randn(8, device='cuda')
    current_view = parameter[1:]
    context = _cpu_offload_context(module)
    storage_keys = tuple(module._module_tensor_registry.tensors_by_storage)
    with context:
        packed_stale_view = context._pack(stale_view)
        packed_current_view = context._pack(current_view)

    assert isinstance(packed_stale_view, _ModuleTensorRef)
    assert isinstance(packed_current_view, _ModuleTensorRef)
    assert not _offloaded_handles(context)
    assert tuple(module._module_tensor_registry.tensors_by_storage) == storage_keys

    parameter.data = full_parameter
    assert torch.equal(packed_stale_view.unpack(), full_parameter.T)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_parallel_module_cpu_offloading_hooks_preserve_zero3_views():
    class TestParallelModule(ParallelModule, skip_init=True):
        def __init__(self):
            ParallelModule.__init__(self)

    weight_data = torch.randn(8, 8, device='cuda')
    module = TestParallelModule()
    module.register_parameter('weight', torch.nn.Parameter(weight_data.clone()))
    module._module_tensor_registry = _ModuleTensorRegistry(module)
    expected_weight = torch.nn.Parameter(weight_data.clone())
    tensor = torch.randn(2, 8, device='cuda', requires_grad=True)
    expected_tensor = tensor.detach().clone().requires_grad_()

    module_tensor_refs = []
    context = module.cpu_offloading_hooks()
    original_pack = context._pack

    def recording_pack(saved_tensor):
        packed = original_pack(saved_tensor)
        if isinstance(packed, _ModuleTensorRef):
            module_tensor_refs.append(packed)
        return packed

    context._pack = recording_pack
    with context:
        output = torch.nn.functional.linear(tensor, module.weight).sin()
    expected_output = torch.nn.functional.linear(expected_tensor, expected_weight).sin()

    # Simulate ZeRO-3 eviction followed by the backward prefetch. Unpack must
    # rebuild the saved weight view from the Parameter's newly gathered storage.
    module.weight.data = torch.empty(8, device='cuda')
    module.weight.data = weight_data.clone()
    output.sum().backward()
    expected_output.sum().backward()

    assert module_tensor_refs
    assert torch.allclose(output, expected_output)
    assert torch.allclose(tensor.grad, expected_tensor.grad)
    assert torch.allclose(module.weight.grad, expected_weight.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_linear_parameter_is_not_copied_by_activation_offload():
    weight_data = torch.randn(64, 64, device='cuda')
    parameter = torch.nn.Parameter(weight_data.clone())
    expected_parameter = torch.nn.Parameter(weight_data.clone())
    tensor = torch.randn(2, 64, device='cuda', requires_grad=True)
    expected_tensor = tensor.detach().clone().requires_grad_()

    module = _TestParallelModule()
    module.register_parameter('weight', parameter)
    context = _cpu_offload_context(module)
    with context:
        output = torch.nn.functional.linear(tensor, parameter).sin()
    expected = torch.nn.functional.linear(expected_tensor, expected_parameter).sin()
    offloaded_numel = sum(
        handle.host_tensor.numel() for handle in _offloaded_handles(context)
    )

    output.sum().backward()
    expected.sum().backward()

    assert offloaded_numel < parameter.numel()
    assert torch.allclose(tensor.grad, expected_tensor.grad)
    assert torch.allclose(parameter.grad, expected_parameter.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_cpu_offload_transfers_are_non_blocking():
    module = _TestParallelModule()
    with _cpu_offload_context(module):
        pass
    shape = (16 * 1024 * 1024,)
    pinned_warmup = torch.empty(shape, pin_memory=True)
    del pinned_warmup
    tensor = torch.randn(shape, device='cuda', requires_grad=True)
    with torch.no_grad():
        output_warmup = tensor.square()
    del output_warmup
    torch.cuda.synchronize()
    torch.cuda._sleep(2_000_000_000)

    context = _cpu_offload_context(module)
    start = time.perf_counter()
    with context:
        output = tensor.square()
    elapsed = time.perf_counter() - start
    handles = _offloaded_handles(context)

    assert handles
    assert all(handle.host_tensor.is_pinned() for handle in handles)
    assert any(not handle.d2h_event.query() for handle in handles), elapsed

    handles[0].prefetch()
    assert any(not handle.h2d_event.query() for handle in handles)

    output.sum().backward()
    assert tensor.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
@pytest.mark.parametrize(
    ('prefetch_level', 'expected_order'),
    [
        (0, [(2, 0)]),
        (1, [(2, 0), (2, 1)]),
        (2, [(2, 0), (2, 1), (1, 1), (1, 0)]),
        (3, [(2, 0), (2, 1), (1, 1), (1, 0), (0, 1), (0, 0)]),
    ],
)
def test_cpu_offload_prefetch_level(prefetch_level, expected_order):
    _OffloadBatch.last_batch = None
    prefetch_order = []
    module = _TestParallelModule()
    contexts = []
    handles_by_batch = []

    for batch_idx in range(3):
        context = _cpu_offload_context(module, prefetch_level=prefetch_level)
        with context:
            handles = [
                context._pack(torch.randn(8, device='cuda')),
                context._pack(torch.randn(8, device='cuda')),
            ]
        contexts.append(context)
        handles_by_batch.append(handles)
        for handle_idx, handle in enumerate(handles):
            original_prefetch = handle.prefetch

            def record_prefetch(
                batch_idx=batch_idx,
                handle_idx=handle_idx,
                handle=handle,
                original_prefetch=original_prefetch,
            ):
                if handle.device_tensor is None:
                    prefetch_order.append((batch_idx, handle_idx))
                original_prefetch()

            handle.prefetch = record_prefetch

    contexts[2]._unpack(handles_by_batch[2][0])

    assert prefetch_order == expected_order


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_default_prefetch_advances_one_batch_per_backward():
    _OffloadBatch.last_batch = None
    prefetch_order = []
    module = _TestParallelModule()
    contexts = []
    handles_by_batch = []

    for batch_idx in range(3):
        context = _cpu_offload_context(module)
        with context:
            handles = [
                context._pack(torch.randn(8, device='cuda')),
                context._pack(torch.randn(8, device='cuda')),
            ]
        contexts.append(context)
        handles_by_batch.append(handles)
        for handle_idx, handle in enumerate(handles):
            original_prefetch = handle.prefetch

            def record_prefetch(
                batch_idx=batch_idx,
                handle_idx=handle_idx,
                handle=handle,
                original_prefetch=original_prefetch,
            ):
                if handle.device_tensor is None:
                    prefetch_order.append((batch_idx, handle_idx))
                original_prefetch()

            handle.prefetch = record_prefetch

    contexts[2]._unpack(handles_by_batch[2][0])

    assert prefetch_order == [(2, 0), (2, 1), (1, 1), (1, 0)]
    assert all(handle.device_tensor is None for handle in handles_by_batch[0])

    contexts[1]._unpack(handles_by_batch[1][0])

    assert prefetch_order[-2:] == [(0, 1), (0, 0)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_triggered_batch_stops_prefetch_at_segment_boundary():
    module = _TestParallelModule()
    contexts = []
    handles = []

    for _ in range(3):
        context = _cpu_offload_context(module)
        with context:
            handle = context._pack(torch.randn(8, device='cuda'))
        contexts.append(context)
        handles.append(handle)

    contexts[1].batch.triggered = True
    contexts[2]._unpack(handles[2])

    assert handles[0].device_tensor is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_cpu_offload_preserves_non_contiguous_layout():
    base = torch.randn(8, 16, device='cuda', requires_grad=True)
    expected_base = base.detach().clone().requires_grad_()
    tensor = base.T
    expected = expected_base.T

    context = _cpu_offload_context(_TestParallelModule())
    with context:
        output = tensor.square()
    expected_output = expected.square()
    handle = _offloaded_handles(context)[0]

    assert handle.host_tensor.stride() == tensor.stride()
    assert handle.unpack().stride() == tensor.stride()

    output.sum().backward()
    expected_output.sum().backward()

    assert torch.allclose(base.grad, expected_base.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_filo_handles_multiple_forward_calls():
    module = _TestParallelModule()

    def forward(tensor):
        context0 = _cpu_offload_context(module)
        with context0:
            hidden = tensor.sin()

        empty_context = _cpu_offload_context(module)
        with empty_context:
            hidden = hidden + 1

        context1 = _cpu_offload_context(module)
        with context1:
            output = hidden.cos()
        return output, context0, empty_context, context1

    tensor0 = torch.randn(8, device='cuda', requires_grad=True)
    tensor1 = torch.randn(8, device='cuda', requires_grad=True)
    output0, context0, empty0, context1 = forward(tensor0)
    output1, context2, empty1, context3 = forward(tensor1)

    assert not _offloaded_handles(empty0)
    assert not _offloaded_handles(empty1)

    output0.sum().backward()
    assert context0.batch.triggered
    assert context1.batch.triggered
    assert not context2.batch.triggered
    assert not context3.batch.triggered
    assert all(
        handle.device_tensor is None
        for handle in _offloaded_handles(context2) + _offloaded_handles(context3)
    )

    output1.sum().backward()
    assert context2.batch.triggered
    assert context3.batch.triggered
    assert tensor0.grad is not None
    assert tensor1.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_repeated_unpack_does_not_advance_filo_stack():
    tensor = torch.randn(8, device='cuda', requires_grad=True)
    module = _TestParallelModule()

    context0 = _cpu_offload_context(module)
    with context0:
        output0 = tensor.square()
    context1 = _cpu_offload_context(module)
    with context1:
        output1 = tensor.sin()

    target = _offloaded_handles(context0)[0]
    context0._unpack(target)
    prefetched_before = tuple(
        handle.device_tensor for handle in _offloaded_handles(context1)
    )
    context0._unpack(target)
    prefetched_after = tuple(
        handle.device_tensor for handle in _offloaded_handles(context1)
    )

    assert prefetched_before == prefetched_after
    assert not context1.batch.triggered
    assert all(device_tensor is None for device_tensor in prefetched_after)
    assert output0 is not None and output1 is not None
