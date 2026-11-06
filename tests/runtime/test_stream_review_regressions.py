# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import OrderedDict, namedtuple
from enum import Enum
from types import SimpleNamespace

import pytest
import torch

import nnscaler
from nnscaler.codegen.schedule.schedule import ScheduleCodeGen
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.runtime import stream as lifetime
from nnscaler.runtime import executor
from nnscaler.runtime.executor import AsyncCommHandler, Executor
from .test_stream_boundaries import streams, pressure, N


@pytest.fixture(autouse=True)
def clean_runtime():
    Executor.clear()
    AsyncCommHandler().clear()
    yield
    Executor.clear()
    AsyncCommHandler().clear()


def test_cpu_metadata_does_not_create_a_cuda_hold(monkeypatch):
    class Mode(Enum):
        TRAIN = 'train'

    metadata = (torch.Size([2, 3]), slice(0, 2), range(3), Mode.TRAIN,
                namedtuple('Shape', 'dims')(torch.Size([2, 3])), OrderedDict(a=4))
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: object())
    monkeypatch.setattr(torch.cuda, 'Event', lambda: pytest.fail('CPU metadata created a CUDA event'))
    with lifetime.input_stream_lifetime(metadata, True) as owners:
        assert not owners
    assert not lifetime._pending_holds


@pytest.mark.parametrize('target', ['executor', 'communication'])
def test_clear_resets_state_even_when_cuda_drain_raises(monkeypatch, target):
    Executor._detach['pending'] = [(1, None)]
    Executor._backward_owners['pending'] = [object()]
    handler = AsyncCommHandler()
    handler._works[1] = [object()]

    def failed_drain(**kwargs):
        raise RuntimeError('CUDA failure')

    with monkeypatch.context() as patch:
        patch.setattr(executor, 'drain_stream_holds', failed_drain)
        with pytest.raises(RuntimeError, match='CUDA failure'):
            (Executor if target == 'executor' else handler).clear()
        if target == 'executor':
            Executor.check_clear()
        else:
            handler.check_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_executor_default_protects_forward_only_inputs():
    producer, consumer = streams()
    with torch.cuda.stream(producer):
        pending = [torch.full((N,), 2., device='cuda')]
    with torch.cuda.stream(consumer):
        weight = torch.nn.Parameter(torch.ones(N, device='cuda'))
        consumer.wait_stream(producer)

        def consume(value):
            torch.cuda._sleep(100_000_000)
            return weight + value

        result = Executor.fexecute('default', consume, pending.pop())
    with torch.cuda.stream(producer):
        replaced = pressure()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.full_like(result, 3), rtol=0, atol=0)
    with torch.cuda.stream(consumer):
        Executor.backward('default', [], [result], [torch.ones_like(result)])
    consumer.synchronize()
    assert replaced
    Executor.check_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_emitted_async_receive_wait_protects_assembly_input():
    producer, consumer = streams()
    value = IRFullTensor((N,), name='recv', dtype=torch.float32).tosub()
    node = IRSegment([], [value], [])
    adapter = IRSegment([], [], [value], name='adapter')
    emitter = ScheduleCodeGen.__new__(ScheduleCodeGen)
    emitter.execplan = SimpleNamespace(use_multi_streams=True)
    code = emitter._emit_async_recv_waits(node, {value.tid: (adapter, value)}, set())
    name = emitter.tensor_name(value)

    def assemble(pending):
        pending = AsyncCommHandler().wait(pending)
        torch.cuda._sleep(100_000_000)
        return torch.cat((pending, pending))

    model = SimpleNamespace(**{emitter.node_name(adapter) + '_wait': assemble})
    namespace = dict(nnscaler=nnscaler, model=model)
    exec('\n'.join([f'def run({name}):', *['    ' + line for line in code],
                    f'    return {name}']), namespace)
    with torch.cuda.stream(producer):
        pending = [torch.full((N,), 2., device='cuda')]
    with torch.cuda.stream(consumer):
        consumer.wait_stream(producer)
        result = namespace['run'](pending.pop())
    with torch.cuda.stream(producer):
        replaced = pressure()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.full_like(result, 2), rtol=0, atol=0)
    assert replaced


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('mode', ['backward', 'split', 'weight_only'])
def test_backward_pre_hook_protects_the_gradient_it_replaces(mode):
    producer, consumer = streams()
    with torch.cuda.stream(producer):
        pending = [torch.full((N,), 2., device='cuda')]
    with torch.cuda.stream(consumer):
        x = torch.ones(N, device='cuda', requires_grad=mode != 'weight_only')
        w = torch.nn.Parameter(torch.ones_like(x))
        y = Executor.fexecute('hook', lambda t: t * w, x, track_stream=True)
        consumer.wait_stream(producer)

        def hook(inputs, outputs, grads):
            torch.cuda._sleep(100_000_000)
            return inputs, outputs, [grads[0] * 2]

        Executor._backward_pre_hook = hook
        if mode == 'backward':
            result = Executor.backward('hook', [x], [y], [pending.pop()], track_stream=True)
        else:
            result = Executor.backward_input('hook', [x], [y], [pending.pop()], [w], track_stream=True)
            Executor.backward_weight('hook', [w], track_stream=True)
            if mode == 'weight_only':
                result = w.grad
    with torch.cuda.stream(producer):
        replaced = pressure()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.full_like(result, 4), rtol=0, atol=0)
    assert replaced
    Executor.check_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_dtensor_uses_the_owner_fallback(tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate

    dist.init_process_group('gloo', init_method=(tmp_path / 'store').as_uri(), rank=0, world_size=1)
    try:
        mesh = init_device_mesh('cuda', (1,))
        value = DTensor.from_local(torch.ones(8, device='cuda'), mesh, [Replicate()])
        with lifetime.input_stream_lifetime((value,), True) as owners:
            assert owners == [value]
        result = Executor.fexecute('wrapped', lambda t: t * 2, value, requires_grad=False)
        torch.testing.assert_close(result.to_local(), torch.full((8,), 2., device='cuda'))
        value.requires_grad_()
        result = Executor.fexecute('wrapped', lambda t: t * 2, value)
        grad = Executor.backward('wrapped', [value], [result], [torch.ones_like(result)])
        torch.testing.assert_close(grad.to_local(), torch.full((8,), 2., device='cuda'))
        lifetime.drain_stream_holds(wait=True)
        Executor.check_clear()
    finally:
        dist.destroy_process_group()
