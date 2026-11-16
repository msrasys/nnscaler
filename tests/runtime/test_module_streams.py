# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import gc
import weakref
from threading import Event

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nnscaler.runtime.stream import ModuleStreams, AsyncTensorCopies, drain_stream_holds


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('pinned', [False, True])
def test_small_copy_does_not_wait_and_snapshots_source(pinned):
    source = torch.arange(16, dtype=torch.int64, pin_memory=pinned)
    expected = source.clone()
    anchor_done = torch.cuda.Event()
    torch.cuda._sleep(500_000_000)
    anchor_done.record()
    with AsyncTensorCopies():
        result = source.to('cuda')
    assert not anchor_done.query(), 'metadata copy waited for the caller stream'
    source.fill_(-1)
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize('grad_enabled', [False, True])
def test_host_submission_is_concurrent_and_preserves_grad_mode(grad_enabled):
    started, release = Event(), Event()
    x = torch.randn(32, device='cuda', requires_grad=True)

    def branch(value):
        started.set()
        assert release.wait(5), 'caller did not continue while branch was submitted'
        return (value.square(),)

    with torch.set_grad_enabled(grad_enabled), ModuleStreams() as streams:
        streams.submit('region', 'test_host', branch, (x,), (x,))
        try:
            assert started.wait(5)
        finally:
            release.set()
        (y,) = streams.resolve('region')
        streams.join('region')
        assert y.requires_grad == grad_enabled
    torch.testing.assert_close(y, x.square())
    if grad_enabled:
        y.sum().backward()
        torch.testing.assert_close(x.grad, 2 * x)


def test_after_launch_does_not_wait_for_anchor_completion():
    x = torch.ones(1024, device='cuda')
    torch.cuda.synchronize()
    anchor_done, side_done = torch.cuda.Event(), torch.cuda.Event()
    with ModuleStreams() as streams:
        streams.capture('side')
        # Model an asynchronous communication wait inserted on the caller.
        torch.cuda._sleep(500_000_000)
        anchor_done.record()
        with streams.fork('side', 'test_after', (x,)):
            y = x + 3
            side_done.record()
        streams.publish((y,))
        side_done.synchronize()
        assert not anchor_done.query(), 'side work serialized behind the anchor'
    torch.testing.assert_close(y, torch.full_like(y, 4))


@pytest.mark.parametrize('recompute', [False, True])
def test_module_stream_forward_backward_matches_serial(recompute):
    torch.manual_seed(1107)
    layers = torch.nn.ModuleList([torch.nn.Linear(128, 256), torch.nn.Linear(256, 128)]).cuda()
    reference = copy.deepcopy(layers)
    x = torch.randn(64, 128, device='cuda', requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()

    def prefix(value):
        return layers[0](value).relu()

    for _ in range(3):
        with ModuleStreams() as streams:
            with streams.fork('prefix', 'test_shared', (x, *layers[0].parameters())):
                h = checkpoint(prefix, x, use_reentrant=False) if recompute else prefix(x)
            streams.publish((h,))
            routed = x.sin()
            with streams.fork('suffix', 'test_shared', (h, *layers[1].parameters())):
                shared = layers[1](h)
            streams.publish((shared,))
            routed = routed.cos()
            streams.join('suffix')
            actual = routed + shared
        expected = ref_x.sin().cos() + reference[1](reference[0](ref_x).relu())
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, ref_x.grad)
        for a, b in zip(layers.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
    torch.cuda.synchronize()
    drain_stream_holds(wait=True)


def test_module_stream_storage_survives_allocator_pressure():
    # Warm allocation bins on both streams before delaying a consumer, so a
    # cudaMalloc does not accidentally synchronize away a use-after-free.
    size = 1 << 20
    with ModuleStreams() as streams:
        with streams.fork('warm', 'test_lifetime', ()):
            warm = [torch.empty(size, device='cuda') for _ in range(16)]
        streams.publish(warm)
    del warm
    torch.cuda.synchronize()
    for _ in range(4):
        x = torch.full((size,), 3., device='cuda')
        with ModuleStreams() as streams:
            with streams.fork('delayed', 'test_lifetime', (x,)):
                torch.cuda._sleep(20_000_000)
                y = x.square()
            streams.publish((y,))
            del x
            garbage = [torch.full((size,), -17., device='cuda') for _ in range(16)]
            streams.join('delayed')
            result = y + 1
            del y
        torch.testing.assert_close(result, torch.full_like(result, 10.))
        del garbage, result
    drain_stream_holds(wait=True)


def test_opaque_input_owner_survives_backward_on_another_stream():
    class Owner:
        def __init__(self):
            self.tensor = torch.full((1 << 20,), 3., device='cuda')

    class UseOwner(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, owner):
            ctx.owner = owner
            return x * owner.tensor

        @staticmethod
        def backward(ctx, grad):
            torch.cuda._sleep(100_000_000)
            result = grad * ctx.owner.tensor
            ctx.owner = None
            return result, None

    owner = Owner()
    reference = weakref.ref(owner)
    producer, consumer = torch.cuda.current_stream(), torch.cuda.Stream()
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer):
        x = torch.ones_like(owner.tensor, requires_grad=True)
        with ModuleStreams() as streams:
            with streams.fork('owner', 'test_owner', (x, owner)):
                y = UseOwner.apply(x, owner)
            streams.publish((y,))
            streams.join('owner')
        drain_stream_holds(wait=True)
        del owner
        y.sum().backward()
        del y
    gc.collect()
    assert reference() is not None
    garbage = [torch.full((1 << 20,), -11., device='cuda') for _ in range(8)]
    torch.cuda.synchronize()
    torch.testing.assert_close(x.grad, torch.full_like(x, 3.))
    drain_stream_holds(wait=True)
    gc.collect()
    assert reference() is None
    del garbage
