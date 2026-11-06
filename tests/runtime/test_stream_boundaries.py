# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from types import SimpleNamespace
import sys
import weakref

import pytest
import torch

import nnscaler
from nnscaler.codegen.lifecycle import LifeCycle
from nnscaler.codegen.schedule.schedule import ScheduleCodeGen
from nnscaler.flags import CompileFlag
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.cten import IRCell, IRObject
from nnscaler.ir.operator import IRFwOperation, IRBpOperation
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.runtime import stream as lifetime
from nnscaler.runtime.adapter import collectives
from nnscaler.runtime.executor import AsyncCommHandler, Executor


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
N = 1 << 20


@pytest.fixture(autouse=True)
def clean_runtime():
    Executor.clear()
    AsyncCommHandler().clear()
    yield
    torch.cuda.synchronize()
    Executor.clear()
    AsyncCommHandler().clear()


def streams():
    producer, consumer = torch.cuda.default_stream(), torch.cuda.Stream()
    # Avoid cudaMalloc synchronization hiding reuse while the consumer is delayed.
    for stream in (producer, consumer):
        with torch.cuda.stream(stream):
            warm = [torch.empty(N, device='cuda') for _ in range(24)]
            warm += [torch.empty(2 * N, device='cuda') for _ in range(8)]
            # First-use CUDA module loading can also synchronize streams.
            warm[0].fill_(2)
            warm[1].copy_(warm[0] + 1)
            warm[2].fill_(-7)
            warm[24].fill_(2)
            warm[25].copy_(warm[24] + 1)
            warm[26].copy_(torch.cat((warm[0], warm[0])))
        stream.synchronize()
        del warm
    return producer, consumer


def pressure():
    return [torch.empty(N, device='cuda').fill_(-7) for _ in range(24)]


@pytest.mark.parametrize('entry', ['forward', 'adapter', 'backward', 'split'])
def test_rdgather_callback_sources_survive_allocation_pressure(monkeypatch, entry):
    producer, consumer = streams()

    class Receive:
        def wait(self):
            torch.cuda.current_stream().wait_stream(producer)
            torch.cuda._sleep(100_000_000)

    def receive(tensor, *args, **kwargs):
        tensor.fill_(2)
        return Receive()

    monkeypatch.setattr(collectives.torch.distributed, 'get_rank', lambda: 0)
    monkeypatch.setattr(collectives.torch.distributed, 'irecv', receive)
    monkeypatch.setattr(collectives, 'DeviceGroup',
                        lambda: SimpleNamespace(get_p2p_group=lambda *args: (None, 0, 0)))
    with torch.cuda.stream(producer):
        pending = [collectives.rdgather(None, (N,), torch.float32, 0, (1, 2), 0, async_op=True)]
    with torch.cuda.stream(consumer):
        x = torch.ones(2 * N, device='cuda', requires_grad=True)
        y = Executor.fexecute('grad', lambda value: value * 1, x, track_stream=True) \
            if entry in ('backward', 'split') else None

    def consume(handle):
        if entry == 'forward':
            return Executor.fexecute('forward', lambda value: value + 1, handle,
                                     requires_grad=False, track_stream=True)
        if entry == 'adapter':
            return Executor.aexecute(lambda value: value + 1, handle,
                                     requires_grad=False, track_stream=True)
        if entry == 'backward':
            return Executor.backward('grad', [x], [y], [handle], track_stream=True)
        result = Executor.backward_input('grad', [x], [y], [handle], [], track_stream=True)
        Executor.backward_weight('grad', [], track_stream=True)
        return result

    with torch.cuda.stream(consumer):
        actual = consume(pending.pop())
    with torch.cuda.stream(producer):
        replacements = pressure()
    torch.cuda.synchronize()
    expected = 3 if entry in ('forward', 'adapter') else 2
    torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=0, atol=0)
    assert replacements
    Executor.check_clear()


@dataclass
class Owner:
    payload: torch.Tensor


def delayed_post_hook(model, meta, inputs, kwargs, output):
    # Make the old forward-only hold eligible for collection before this work.
    torch.cuda.current_stream().synchronize()
    torch.cuda._sleep(500_000_000)
    model.hook_result = inputs[0].payload + 1


def compile_run(lines):
    namespace = dict(torch=torch, nnscaler=nnscaler, pressure=pressure, lifetime=lifetime)
    root_module = __name__.split('.')[0]
    namespace[root_module] = sys.modules[root_module]
    exec('\n'.join(lines), namespace)
    return namespace['run']


def test_emitted_post_hook_keeps_opaque_input_until_gpu_completion():
    producer, consumer = streams()
    metadata = IRObject('metadata')
    output = IRFullTensor((1,), name='output', dtype=torch.float32).tosub()
    node = IRSegment([], [metadata], [output])
    node.post_hook = delayed_post_hook
    emitter = ScheduleCodeGen.__new__(ScheduleCodeGen)
    emitter.execplan = SimpleNamespace(use_multi_streams=True)
    codes = emitter.emit_node(node)
    release = emitter.emit_release(LifeCycle([node], [metadata], [output]).release_tensors_after_node(node))
    run = compile_run([
        f'def run(model, {emitter.tensor_name(metadata)}):',
        '    with torch.cuda.stream(model.consumer):',
        *['        ' + code for code in codes],
        '    ' + release,
        '    with torch.cuda.stream(model.producer):',
        '        replacements = nnscaler.runtime.executor.aexecute(pressure, requires_grad=False, track_stream=True)',
        '    return replacements',
    ])
    model = SimpleNamespace(producer=producer, consumer=consumer)
    setattr(model, emitter.node_name(node), lambda _: torch.empty(0, device='cuda'))
    with torch.cuda.stream(producer):
        owner = Owner(torch.full((N,), 2., device='cuda'))
    reference = weakref.ref(owner)
    pending = [owner]
    del owner
    consumer.wait_stream(producer)
    replacements = run(model, pending.pop())
    torch.cuda.synchronize()
    torch.testing.assert_close(model.hook_result, torch.full_like(model.hook_result, 3), rtol=0, atol=0)
    lifetime.drain_stream_holds(wait=True)
    assert reference() is None and replacements


class HiddenScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, owner):
        ctx.owner = owner
        return value * owner.payload

    @staticmethod
    def backward(ctx, grad):
        torch.cuda._sleep(100_000_000)
        return grad * ctx.owner.payload, None


@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('gradients', ['input', 'weight', 'both'])
@pytest.mark.parametrize('backward_on_producer', [False, True])
def test_emitted_backward_keeps_opaque_autograd_dependency(monkeypatch, split, gradients, backward_on_producer):
    producer, consumer = streams()
    monkeypatch.setattr(CompileFlag, 'use_fbw', split)

    def tensor(name, requires_grad=True):
        value = IRFullTensor((N,), name=name, dtype=torch.float32, requires_grad=requires_grad).tosub()
        if requires_grad:
            value.grad = value.parent.grad.tosub()
        return value

    x, output = tensor('x', gradients != 'weight'), tensor('output')
    metadata = IRObject('metadata')
    op = IRFwOperation('scale', 'scale', [x, metadata], 1)
    op.set_output(0, output)
    grads = [x.grad] if x.requires_grad else []
    bp = IRBpOperation([output.grad], grads)
    forward = IRSegment([op], [x, metadata], [output])
    backward = IRSegment([bp], [output.grad], grads)
    IRCell.make_pair(forward, backward)
    emitter = ScheduleCodeGen.__new__(ScheduleCodeGen)
    emitter.execplan = SimpleNamespace(use_multi_streams=True, outputs=lambda: [])
    lifetime_plan = LifeCycle([forward, backward], [x, metadata, output.grad], grads)
    forward_codes = emitter.emit_node(forward)
    forward_codes += [emitter.emit_release(lifetime_plan.release_tensors_after_node(forward))]
    forward_codes += ['torch.cuda.current_stream().synchronize()', 'lifetime.drain_stream_holds()']
    backward_codes = emitter.emit_node(backward, produced_tids={output.grad.tid})
    backward_codes += [emitter.emit_release(lifetime_plan.release_tensors_after_node(backward))]
    result = emitter.tensor_name(x.grad) if x.requires_grad else 'None'
    run = compile_run([
        f'def run(model, {emitter.tensor_name(x)}, {emitter.tensor_name(metadata)}, {emitter.tensor_name(output.grad)}):',
        '    with torch.cuda.stream(model.consumer):',
        *['        ' + code for code in forward_codes],
        '    with torch.cuda.stream(model.backward_stream):',
        *['        ' + code for code in backward_codes],
        '    with torch.cuda.stream(model.producer):',
        '        replacements = nnscaler.runtime.executor.aexecute(pressure, requires_grad=False, track_stream=True)',
        f'    return {result}, replacements',
    ])
    with torch.cuda.stream(producer):
        owner = Owner(torch.full((N,), 2., device='cuda'))
    reference = weakref.ref(owner)
    pending = [owner]
    del owner
    with torch.cuda.stream(consumer):
        value = torch.ones(N, device='cuda', requires_grad=gradients != 'weight')
        weight = torch.nn.Parameter(torch.ones(N, device='cuda'))
        upstream = torch.ones_like(value)
    model = SimpleNamespace(producer=producer, consumer=consumer,
                            backward_stream=producer if backward_on_producer else consumer,
                            parameters=lambda: [] if gradients == 'input' else [weight])

    def body(value, owner):
        if gradients == 'input':
            return HiddenScale.apply(value, owner)
        return value * HiddenScale.apply(weight, owner)

    setattr(model, emitter.node_name(forward), body)
    consumer.wait_stream(producer)
    actual, replacements = run(model, value, pending.pop(), upstream)
    torch.cuda.synchronize()
    if gradients != 'weight':
        torch.testing.assert_close(actual, torch.full_like(actual, 2), rtol=0, atol=0)
    if gradients != 'input':
        torch.testing.assert_close(weight.grad, torch.full_like(weight, 2), rtol=0, atol=0)
    lifetime.drain_stream_holds(wait=True)
    assert reference() is None and replacements
    Executor.check_clear()
