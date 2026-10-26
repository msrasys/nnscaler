#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch

from nnscaler.runtime import cpu_offloading
from nnscaler.runtime.cpu_offloading import (
    CPUOffloadContext,
    OffloadManager,
    _ModuleTensorRef,
    _OffloadedTensor,
    _PinnedPool,
)
from nnscaler.runtime.module import _ModuleTensorRegistry, ParallelModule


class _Host(torch.nn.Module):
    """Minimal stand-in for a ParallelModule: offloading only needs the owner lookup."""

    def __init__(self, width=256, depth=3):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(width, width) for _ in range(depth)])
        self._registry = _ModuleTensorRegistry(self)

    def find_buffer_or_param_owner(self, tensor):
        return self._registry.find(tensor)


@pytest.fixture(autouse=True)
def _fresh_manager(monkeypatch):
    monkeypatch.setenv('CPU_OFFLOADING_MIN_BYTES', '1024')
    OffloadManager.reset()
    yield
    OffloadManager.reset()


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')


def _segment_forward(host, x, name, manager, group_ids=None):
    ctx = manager.begin_forward(name)
    h = x
    try:
        for i, layer in enumerate(host.layers):
            with CPUOffloadContext(host, group_id=i if group_ids is None else group_ids[i]):
                h = torch.nn.functional.gelu(layer(h))
    finally:
        manager.end_forward(ctx)
    return h, ctx


def _segment_backward(loss, name, manager):
    ctx = manager.begin_backward(name)
    try:
        loss.backward()
    finally:
        manager.end_backward(ctx)
    return ctx


@cuda_only
def test_gradients_match_without_offload():
    torch.manual_seed(0)
    host = _Host().cuda()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)

    def run(offload):
        manager = OffloadManager.get()
        host.zero_grad(); x.grad = None
        if offload:
            h, _ = _segment_forward(host, x, 'seg', manager)
            loss = h.float().square().mean()
            _segment_backward(loss, 'seg', manager)
        else:
            h = x
            for layer in host.layers:
                h = torch.nn.functional.gelu(layer(h))
            loss = h.float().square().mean()
            loss.backward()
        return loss.detach().clone(), x.grad.clone(), [p.grad.clone() for p in host.parameters()]

    ref = run(False)
    for _ in range(2):
        out = run(True)
        assert torch.equal(out[0], ref[0])
        assert torch.equal(out[1], ref[1])
        assert all(torch.equal(a, b) for a, b in zip(out[2], ref[2]))
    stats = cpu_offloading.offload_stats()
    assert stats['offloaded_regions'] > 0
    assert stats['pending_contexts'] == 0


@cuda_only
def test_parameters_are_not_offloaded():
    host = _Host().cuda()
    manager = OffloadManager.get()
    ctx = manager.begin_forward('seg')
    context = CPUOffloadContext(host)
    with context:
        packed = context._pack(host.layers[0].weight)
        activation = torch.randn(64, 256, device='cuda')
        packed_act = context._pack(activation)
    manager.end_forward(ctx)
    assert isinstance(packed, _ModuleTensorRef)
    assert isinstance(packed_act, _OffloadedTensor)
    assert torch.equal(packed.unpack(), host.layers[0].weight)


@cuda_only
def test_views_share_one_region_and_are_rebuilt_with_strides():
    host = _Host().cuda()
    manager = OffloadManager.get()
    base = torch.randn(8, 1024, device='cuda')
    ctx = manager.begin_forward('seg')
    context = CPUOffloadContext(host)
    with context:
        a = context._pack(base[:, :512])          # non-contiguous view
        b = context._pack(base[:, 512:])          # disjoint view, same storage
        c = context._pack(base.t())               # transposed view of everything
    manager.end_forward(ctx)
    batch = context.batch
    # a and b cover different byte ranges; c covers the whole storage
    assert len(batch.regions) == 3 or len(batch.regions) == 2
    manager.begin_backward('seg')
    torch.cuda.synchronize()
    assert torch.equal(context._unpack(a), base[:, :512])
    assert torch.equal(context._unpack(b), base[:, 512:])
    assert torch.equal(context._unpack(c), base.t())


@cuda_only
def test_pinned_pool_is_reused_with_varying_sizes():
    host = _Host().cuda()
    manager = OffloadManager.get()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    sizes_after = []
    for step in range(4):
        # vary the token count every step like MoE routing does
        xs = x[: 40 + step * 5]
        h, _ = _segment_forward(host, xs, 'seg', manager)
        _segment_backward(h.float().sum(), 'seg', manager)
        sizes_after.append((manager.pool.allocations, manager.pool.allocated_bytes))
    # the first step allocates, later steps reuse size classes
    assert sizes_after[-1][0] <= sizes_after[0][0] + len(host.layers)
    assert _PinnedPool.size_class(1) == _PinnedPool._MIN_CLASS
    assert _PinnedPool.size_class(_PinnedPool._MIN_CLASS + 1) > _PinnedPool._MIN_CLASS


@cuda_only
def test_contexts_are_consumed_fifo_per_segment():
    host = _Host().cuda()
    manager = OffloadManager.get()
    x0 = torch.randn(64, 256, device='cuda', requires_grad=True)
    x1 = torch.randn(64, 256, device='cuda', requires_grad=True)
    h0, ctx0 = _segment_forward(host, x0, 'seg', manager)
    h1, ctx1 = _segment_forward(host, x1, 'seg', manager)
    assert manager.snapshot(reset=False)['pending_contexts'] == 2
    popped = manager.begin_backward('seg')
    assert popped is ctx0
    # the last group of the oldest context is prefetched first
    assert ctx0.batches[-1].prefetched
    assert not ctx1.batches[-1].prefetched
    manager.prefetch_next_backward('seg')
    assert ctx1.batches[-1].prefetched
    h0.float().sum().backward()
    manager.end_backward(popped)
    assert manager.begin_backward('seg') is ctx1
    h1.float().sum().backward()
    manager.end_backward(ctx1)
    assert manager.snapshot()['pending_contexts'] == 0


@cuda_only
def test_hint_forward_disables_offload_for_immediate_backward():
    host = _Host().cuda()
    manager = OffloadManager.get()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    cpu_offloading.hint_forward('seg', 0)
    h, ctx = _segment_forward(host, x, 'seg', manager)
    assert not ctx.offload_enabled and not ctx.batches
    assert manager.snapshot(reset=False)['pending_contexts'] == 0
    h.float().sum().backward()
    # a far away backward keeps offloading on
    cpu_offloading.hint_forward('seg', 5)
    h, ctx = _segment_forward(host, x, 'seg', manager)
    assert ctx.offload_enabled and ctx.batches
    _segment_backward(h.float().sum(), 'seg', manager)


@cuda_only
def test_eager_chain_prefetch_without_executor():
    host = _Host().cuda()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    h = x
    contexts = []
    for i, layer in enumerate(host.layers):
        context = CPUOffloadContext(host, group_id=i)
        with context:
            h = torch.nn.functional.gelu(layer(h))
        contexts.append(context)
    assert all(c.batch.context is None for c in contexts)
    h.float().sum().backward()
    # once the last batch was demanded, its predecessor was prefetched
    assert contexts[-2].batch.prefetched


def test_disable_env_turns_contexts_into_noops(monkeypatch):
    monkeypatch.setenv('CPU_OFFLOADING_DISABLE', '1')
    OffloadManager.reset()
    if not torch.cuda.is_available():
        pytest.skip('lack of gpu devices')
    host = _Host().cuda()
    context = CPUOffloadContext(host)
    with context:
        assert context._hooks is None


def test_parallel_module_cpu_offloading_hooks_accepts_group_id():
    class _Module(ParallelModule, skip_init=True):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self._module_tensor_registry = None
            self.cpu_offloading_prefetch_level = 0

    context = _Module().cpu_offloading_hooks(group_id=7)
    assert isinstance(context, CPUOffloadContext)
    assert context.group_id == 7
