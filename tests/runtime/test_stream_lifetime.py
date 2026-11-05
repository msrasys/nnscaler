# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace
import gc
import weakref

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import nnscaler
from nnscaler.codegen.lifecycle import LifeCycle
from nnscaler.codegen.schedule.schedule import ScheduleCodeGen
from nnscaler.graph import IRGraph
from nnscaler.graph.function.pyfunc import IRPyFunc
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.schedule.schedplan import StreamContext
from nnscaler.ir.cten import IRObject
from nnscaler.ir.operator import IRDataOperation
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.runtime.executor import AsyncCommHandler, Executor
from nnscaler.runtime import stream as lifetime


@pytest.fixture(autouse=True)
def clear_runtime():
    Executor.clear()
    AsyncCommHandler().clear()
    yield
    Executor.clear()
    AsyncCommHandler().clear()


def test_disabled_tracking_does_not_visit_inputs(monkeypatch):
    class Owner:
        def record_stream(self, stream):
            pytest.fail('single-stream input should not be inspected')

    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: pytest.fail('CUDA queried'))
    with lifetime.input_stream_lifetime([Owner()], False):
        pass


def emitted_forward(inference, default_consumer, multi_streams=True):
    metadata = IRObject('metadata')
    output = IRFullTensor((1,), name='output', dtype=torch.float32, requires_grad=True).tosub()
    node = IRSegment([], [metadata], [output])
    if not default_consumer:
        node.set_op_context('stream_context', StreamContext(stream='compute'))
    emitter = ScheduleCodeGen.__new__(ScheduleCodeGen)
    emitter.execplan = SimpleNamespace(use_multi_streams=multi_streams)
    codes = emitter.emit_node(node, force_no_grad=inference)
    release = emitter.emit_release(
        LifeCycle([node], [metadata], [output]).release_tensors_after_node(node))
    name = emitter.tensor_name(metadata)
    source = '\n'.join([
        f'def run(model, {name}):',
        '    with torch.cuda.stream(model.consumer):',
        *['        ' + code for code in codes],
        '    ' + release,
        '    with torch.cuda.stream(model.producer):',
        '        replacement = torch.empty(model.n, device="cuda")',
        '        replacement.fill_(-7)',
        '    return ' + emitter.tensor_name(output) + ', replacement',
    ])
    namespace = dict(torch=torch, nnscaler=nnscaler)
    exec(source, namespace)
    return namespace['run'], emitter.node_name(node), source


def test_default_consumer_in_multi_stream_plan_enables_runtime_tracking():
    assert 'track_stream=True' in emitted_forward(False, True)[2]
    assert 'track_stream=' not in emitted_forward(False, True, False)[2]


def test_stream_tracking_consumes_replayed_metadata_after_it_is_defined():
    import operator

    sample, metadata = IRObject('sample'), IRObject('metadata')
    loader = IRObject('dataloader')
    output = IRFullTensor((1,), name='output', dtype=torch.float32).tosub()
    node = IRSegment([], [sample, metadata], [output])
    emitter = ScheduleCodeGen.__new__(ScheduleCodeGen)
    lookup = IRPyFunc('_operator.getitem', [sample, 'meta'], [metadata])
    emitter.execplan = SimpleNamespace(
        use_multi_streams=True,
        graph=IRGraph([IRDataOperation(loader, (sample,)), lookup], [sample], [], 'stream_metadata'),
    )
    planned = emitter._plan_sample_reads([node])[node]
    code = emitter.emit_node(
        node, produced_tids={sample.tid}, sample_reads=planned,
    )
    observed = []

    def forward(name, function, *inputs, **kwargs):
        assert inputs[1] is inputs[0]['meta']
        assert kwargs['track_stream']
        observed.append(inputs[1])
        return torch.zeros(1)

    owner = SimpleNamespace()
    setattr(owner, emitter.node_name(node), None)
    namespace = {
        '_operator': operator, 'model': owner,
        emitter.tensor_name(sample): {'meta': {'payload': torch.ones(1)}},
        'nnscaler': SimpleNamespace(runtime=SimpleNamespace(executor=SimpleNamespace(fexecute=forward))),
    }
    exec('\n'.join(code), namespace)
    assert len(observed) == 1


@dataclass
class OpaqueOwner:
    payload: torch.Tensor


class ProtocolOwner(OpaqueOwner):
    def record_stream(self, stream):
        self.payload.record_stream(stream)


class DictOwner(dict):
    def __init__(self, value):
        super().__init__()
        self.payload = value


class SlotsOwner:
    __slots__ = ('payload',)

    def __init__(self, value):
        self.payload = value


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('inference', [False, True])
@pytest.mark.parametrize('default_consumer', [False, True])
@pytest.mark.parametrize('container', ['nested', 'namedtuple', 'opaque', 'protocol', 'dict_subclass', 'slots'])
def test_generated_forward_releases_container_without_reusing_pending_storage(
    inference, default_consumer, container,
):
    run, name, _ = emitted_forward(inference, default_consumer)
    producer, consumer = torch.cuda.default_stream(), torch.cuda.Stream()
    if default_consumer:
        producer, consumer = consumer, producer
    n = 1 << 20
    # Warm allocation before queuing the delayed consumer. A fresh cudaMalloc
    # can synchronize streams and conceal premature reuse in the broken code.
    with torch.cuda.stream(producer):
        warm = [torch.empty(n, device='cuda') for _ in range(8)]
    torch.cuda.synchronize()
    del warm

    def consume(metadata):
        torch.cuda._sleep(100_000_000)
        payload = metadata['values'][0][0] if container == 'nested' else metadata.payload
        return weight + payload

    for _ in range(3):
        with torch.cuda.stream(producer):
            weight = torch.nn.Parameter(torch.ones(n, device='cuda'))
            value = torch.full((n,), 2., device='cuda')
        reference = weakref.ref(value)
        if container == 'nested':
            metadata = {'values': [(value,)], 'cpu': torch.zeros(1)}
        elif container == 'namedtuple':
            metadata = namedtuple('Metadata', 'payload')(value)
        elif container == 'opaque':
            metadata = OpaqueOwner(value)
        elif container == 'dict_subclass':
            metadata = DictOwner(value)
        elif container == 'slots':
            metadata = SlotsOwner(value)
        else:
            metadata = ProtocolOwner(value)
        model = SimpleNamespace(n=n, producer=producer, consumer=consumer)
        setattr(model, name, consume)
        consumer.wait_stream(producer)
        arguments = [metadata]
        del metadata, value
        with torch.inference_mode(inference):
            output, replacement = run(model, arguments.pop())
        if container in ('nested', 'protocol'):
            assert reference() is None  # protection does not pin the Python tensor
        consumer.synchronize()
        producer.synchronize()
        torch.testing.assert_close(output, torch.full_like(output, 3), rtol=0, atol=0)
        if not inference:
            with torch.cuda.stream(consumer):
                Executor.backward(name, [], [output], [torch.ones_like(output)], track_stream=True)
            consumer.synchronize()
            torch.testing.assert_close(weight.grad, torch.ones_like(weight))
        lifetime.drain_stream_holds(wait=True)
        assert reference() is None
        del output, replacement, weight
        Executor.check_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_nested_cycles_aliases_and_cpu_tensors_are_handled_once(monkeypatch):
    calls = []
    original = torch.Tensor.record_stream

    def record(tensor, stream):
        calls.append(id(tensor))
        return original(tensor, stream)

    monkeypatch.setattr(torch.Tensor, 'record_stream', record)
    value = torch.ones(8, device='cuda')
    metadata = {value: [value, (value,)], 'cpu': torch.ones(8)}
    metadata['cycle'] = metadata
    with lifetime.input_stream_lifetime(metadata, True):
        pass
    assert calls == [id(value)]
    metadata.clear()
    gc.collect()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('adapter', [False, True])
def test_async_replacement_is_recorded_on_the_actual_consumer(monkeypatch, adapter):
    placeholder = torch.ones(4, device='cuda')
    replacement = torch.zeros_like(placeholder)
    calls = []
    original = torch.Tensor.record_stream

    def record(tensor, stream):
        calls.append((tensor.data_ptr(), stream.cuda_stream))
        return original(tensor, stream)

    monkeypatch.setattr(torch.Tensor, 'record_stream', record)
    class Work:
        def wait(self):
            pass

    AsyncCommHandler().submit(placeholder, [Work()], lambda _: replacement)
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(consumer):
        function = lambda tensor: tensor + 3
        if adapter:
            output = Executor.aexecute(function, placeholder, requires_grad=False, track_stream=True)
        else:
            output = Executor.fexecute('replacement', function, placeholder,
                                       requires_grad=False, track_stream=True)
    consumer.synchronize()
    assert calls == [(placeholder.data_ptr(), consumer.cuda_stream),
                     (replacement.data_ptr(), consumer.cuda_stream)]
    torch.testing.assert_close(output, torch.full_like(output, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('split', [False, True])
@pytest.mark.parametrize('recompute', [False, True])
@pytest.mark.parametrize('input_grad', [False, True])
def test_stream_boundaries_preserve_checkpoint_and_split_gradients(split, recompute, input_grad):
    torch.manual_seed(7)
    x = torch.randn(8, 8, device='cuda', requires_grad=input_grad)
    weight = torch.nn.Parameter(torch.randn_like(x))
    metadata = {'bias': torch.randn_like(x)}
    expected_x = x.detach().clone().requires_grad_(input_grad)
    expected_weight = weight.detach().clone().requires_grad_()
    expected = (expected_x @ expected_weight + metadata['bias']).t()
    expected.square().sum().backward()

    def body(value, context):
        return (value @ weight + context['bias']).t()

    def forward(value, context):
        return checkpoint(body, value, context, use_reentrant=False) if recompute else body(value, context)

    compute = torch.cuda.Stream()
    compute.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(compute):
        output = Executor.fexecute('gradients', forward, x, metadata, track_stream=True)
    # Test the implicit default stream as the backward caller as well.
    torch.cuda.current_stream().wait_stream(compute)
    requested = [x] if input_grad else []
    upstream = 2 * output.detach()
    if split:
        grad = Executor.backward_input('gradients', requested, [output], [upstream],
                                       [weight], track_stream=True)
        Executor.backward_weight('gradients', [weight], track_stream=True)
    else:
        grad = Executor.backward('gradients', requested, [output], [upstream], track_stream=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(weight.grad, expected_weight.grad)
    if input_grad:
        torch.testing.assert_close(grad, expected_x.grad)
    Executor.check_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('fail', [False, True])
def test_opaque_owners_survive_until_event_even_on_exception(fail):
    owner = OpaqueOwner(torch.ones(8, device='cuda'))
    ref = weakref.ref(owner)
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())

    def consume(value):
        with torch.cuda.stream(consumer), lifetime.input_stream_lifetime([value], True):
            torch.cuda._sleep(100_000_000)
            value.payload.add_(1)
            if fail:
                raise ValueError('test exception')

    try:
        consume(owner)
    except ValueError:
        assert fail
    del owner
    assert ref() is not None
    lifetime.drain_stream_holds(wait=True)
    assert ref() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_sparse_cuda_metadata_uses_event_hold():
    value = torch.eye(4, device='cuda').to_sparse()
    reference = weakref.ref(value)
    with lifetime.input_stream_lifetime({'sparse': value}, True):
        output = value.to_dense()
    del value
    assert reference() is not None
    lifetime.drain_stream_holds(wait=True)
    assert reference() is None
    torch.testing.assert_close(output, torch.eye(4, device='cuda'))
