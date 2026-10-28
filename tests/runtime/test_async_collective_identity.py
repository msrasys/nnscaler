# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import gc
import json
from types import SimpleNamespace
import weakref

import pytest
import torch

from nnscaler.runtime.adapter import collectives
from nnscaler.runtime.executor import AsyncCommHandler, Executor
from tests.launch_torchrun import torchrun
from tests.runtime.test_runtime_collectives import _init_distributed


def _layout(kind, full, rank):
    idim, odim = (1, 0) if kind == 'single' else (0, 1)
    source = full.chunk(2, dim=idim)[rank].contiguous().detach().requires_grad_()
    expected = full if kind == 'gather' else full.chunk(2, dim=odim)[rank]
    return source, expected, idim, odim


def _start(kind, source, idim, odim):
    if kind == 'gather':
        return collectives.all_gather(source, idim, (0, 1), async_op=True)
    operation = collectives.all_to_all_single if kind == 'single' else collectives.all_to_all
    return operation(source, idim, odim, (0, 1), async_op=True)


@pytest.fixture
def transport(monkeypatch):
    full = torch.arange(1, 25, dtype=torch.float64).reshape(4, 6)
    state = SimpleNamespace(full=full, waits=0, idim=0, odim=1)
    monkeypatch.setattr(collectives, 'DeviceGroup', lambda: SimpleNamespace(get_group=lambda ranks: None))
    monkeypatch.setattr(collectives, 'CudaTimer', lambda: SimpleNamespace(start=lambda **kw: None, stop=lambda **kw: None))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda group=None: 0)
    monkeypatch.setattr(torch.distributed, 'get_process_group_ranks', lambda group: [0, 1])

    def work(outputs, values):
        def wait():
            state.waits += 1
            with torch.no_grad():
                for output, value in zip(outputs, values):
                    output.copy_(value)
        return SimpleNamespace(wait=wait)

    def gather(outputs, source, **kwargs):
        return work(outputs, full.chunk(2, dim=state.idim))

    def alltoall(outputs, inputs, **kwargs):
        values = [part.chunk(2, dim=state.odim)[0] for part in full.chunk(2, dim=state.idim)]
        return work(outputs, values)

    def single(output, source, **kwargs):
        outputs = list(output.chunk(2))
        if state.odim:
            outputs = [part.transpose(0, state.odim) for part in outputs]
        return alltoall(outputs, [], **kwargs)

    monkeypatch.setattr(torch.distributed, 'all_gather', gather)
    monkeypatch.setattr(torch.distributed, 'all_to_all', alltoall)
    monkeypatch.setattr(torch.distributed, 'all_to_all_single', single)
    AsyncCommHandler().clear()
    Executor.clear()
    yield state
    AsyncCommHandler().clear()
    Executor.clear()


@pytest.mark.parametrize('kind', ['gather', 'list', 'single'])
def test_async_result_identity_does_not_capture_producer_branch(transport, kind):
    source, expected, transport.idim, transport.odim = _layout(kind, transport.full, 0)
    original = source.detach().clone()
    handle = _start(kind, source, transport.idim, transport.odim)
    assert handle is not source
    assert handle.data_ptr() == source.data_ptr(), 'a placeholder must not copy the payload'
    assert handle.requires_grad == source.requires_grad

    # A different consumer of the producer must not resolve to this result.
    resolved = Executor.sync_tensors([source, handle, handle])
    torch.testing.assert_close(resolved[0], original)
    torch.testing.assert_close(resolved[1], expected)
    assert resolved[1] is resolved[2]
    assert transport.waits == 1
    torch.testing.assert_close(source, original)
    AsyncCommHandler().check_clear()


@pytest.mark.parametrize('kind', ['gather', 'list', 'single'])
def test_async_registry_retains_placeholder_until_wait(transport, kind):
    source, expected, transport.idim, transport.odim = _layout(kind, transport.full, 0)
    handle = _start(kind, source, transport.idim, transport.odim)
    pointer = handle.data_ptr()
    held = weakref.ref(handle)
    del handle, source
    gc.collect()
    assert held() is not None
    assert held().data_ptr() == pointer
    actual = AsyncCommHandler().wait(held())
    torch.testing.assert_close(actual, expected)
    assert transport.waits == 1
    AsyncCommHandler().check_clear()
    gc.collect()
    assert held() is None


@pytest.mark.parametrize('kind', ['gather', 'list', 'single'])
def test_producer_and_received_boundaries_keep_separate_gradients(transport, kind):
    source, expected, transport.idim, transport.odim = _layout(kind, transport.full, 0)
    handle = _start(kind, source, transport.idim, transport.odim)
    callbacks = []
    Executor.register_input_grad_callback(source, lambda grad: callbacks.append(('source', grad)))
    Executor.register_input_grad_callback(handle, lambda grad: callbacks.append(('result', grad)))
    local = Executor.fexecute('local', lambda x: x.square().sum(), source)
    assert transport.waits == 0
    remote = Executor.fexecute('remote', lambda x, y: (x * y).sum(), handle, handle)
    local_grad = Executor.backward('local', [source], [local], [torch.ones_like(local)])
    remote_grads = Executor.backward('remote', [handle, handle], [remote], [torch.ones_like(remote)])
    torch.testing.assert_close(local_grad, 2 * source)
    for grad in remote_grads:
        torch.testing.assert_close(grad, 2 * expected)
    assert [name for name, _ in callbacks] == ['source', 'result']
    assert transport.waits == 1
    Executor.check_clear()
    AsyncCommHandler().check_clear()


def _identity_worker(kind):
    _init_distributed(2)
    rank = torch.distributed.get_rank()
    full = torch.arange(1, 25, dtype=torch.float64).reshape(4, 6)
    reference = full.detach().clone().requires_grad_()
    source, expected, idim, odim = _layout(kind, full, rank)
    handle = _start(kind, source, idim, odim)
    assert handle is not source and handle.data_ptr() == source.data_ptr()
    assert handle.requires_grad
    # This source consumer must run without waiting for the distinct result.
    local_loss = Executor.fexecute('local', lambda x: x.square().sum(), source)
    result_loss = Executor.fexecute('result', lambda x: x.square().sum(), handle)
    expected_local_loss = source.detach().square().sum()
    expected_result_loss = expected.square().sum()
    torch.testing.assert_close(local_loss, expected_local_loss)
    torch.testing.assert_close(result_loss, expected_result_loss)
    local_grad = Executor.backward('local', [source], [local_loss], [torch.ones_like(local_loss)])
    result_grad = Executor.backward('result', [handle], [result_loss], [torch.ones_like(result_loss)])
    if kind == 'gather':
        reverse = collectives.reduce_scatter(result_grad, idim, (0, 1), async_op=True)
        reference_loss = sum(part.square().sum() for part in reference.chunk(2, dim=idim)) + 2 * reference.square().sum()
    else:
        operation = collectives.all_to_all_single if kind == 'single' else collectives.all_to_all
        reverse = operation(result_grad, odim, idim, (0, 1), async_op=True)
        reference_loss = (sum(part.square().sum() for part in reference.chunk(2, dim=idim))
                          + sum(part.square().sum() for part in reference.chunk(2, dim=odim)))
    actual_grad = local_grad + AsyncCommHandler().wait(reverse)
    reference_loss.backward()
    expected_grad = reference.grad.chunk(2, dim=idim)[rank]
    torch.testing.assert_close(actual_grad, expected_grad)
    actual_optimizer = torch.optim.SGD([source], lr=0.01)
    reference_optimizer = torch.optim.SGD([reference], lr=0.01)
    source.grad = actual_grad
    actual_optimizer.step()
    reference_optimizer.step()
    expected_weight = reference.chunk(2, dim=idim)[rank]
    torch.testing.assert_close(source, expected_weight)
    # Measure only after assertions so diagnostics cannot hide a missing wait.
    print('NNSCALER_TEST_METRICS ' + json.dumps({
        'case': 'async_producer_identity', 'kind': kind, 'rank': rank,
        'loss_max_abs': max((local_loss - expected_local_loss).abs().item(),
                            (result_loss - expected_result_loss).abs().item()),
        'gradient_max_abs': (actual_grad - expected_grad).abs().max().item(),
        'weight_max_abs': (source - expected_weight).abs().max().item(),
    }))
    Executor.check_clear()
    AsyncCommHandler().check_clear()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires 2 GPUs')
@pytest.mark.parametrize('kind', ['gather', 'list', 'single'])
def test_async_producer_identity_nccl(kind):
    torchrun(2, _identity_worker, kind)
